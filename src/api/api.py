import sys
sys.path.append("src")
import torch
import open_clip
from datetime import datetime
from PIL import Image
from conf import config
from models.evaluate import predict_context,predict
from api.schemas import PredictContextPayload,PredictImagesPayload
from http import HTTPStatus
from pydantic import ValidationError
from fastapi import FastAPI, HTTPException, status, Request, File, UploadFile, Depends, Form
from fastapi.encoders import jsonable_encoder
from typing import List
from io import BytesIO

dev = "cuda" if torch.cuda.is_available() else "cpu"
__pretrain_models = {"RN50": "openai",
                     "ViT-B-16": "laion2b_s34b_b88k"}

model_dict = {}
preproc = open_clip.image_transform(224, False)

app = FastAPI(
    title="Image Text disambiguation APIs",
    version="1.0"
)

@app.on_event("startup")
def _load_models_and_transformation():
    # Load models
    for model_name in __pretrain_models:
        model = open_clip.create_model(model_name,
                                       __pretrain_models[model_name],
                                       dev)
        if model_name == "RN50":
            model.load_state_dict(torch.load(config["MODEL_FILE"],
                                             map_location=dev))
        model_dict[model_name] = model

def checker_context(data: str = Form(...)):
    try:
        return PredictContextPayload.model_validate_json(data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            detail=jsonable_encoder(e.errors()),
        )

def checker_images(data: str = Form(...)):
    try:
        return PredictImagesPayload.model_validate_json(data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            detail=jsonable_encoder(e.errors()),
        )

def construct_response(request: Request, response: dict):
    final_response = {
        "message": response["message"],
        "method": request.method,
        "status-code": response["status-code"],
        "timestamp": datetime.now().isoformat(),
        "url": request.url._url
    }
    if "data" in response:
        final_response["data"] = response["data"]

    return final_response

@app.post("/models/{model_name}/predict_context")
def _predict_context(request: Request, model_name: str, payload: PredictContextPayload = Depends(checker_context), image: UploadFile = File(...)):

    if model_name not in model_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not found")

    word = payload.target_word
    contexts = payload.contexts
    image = preproc(Image.open(image.file))

    scores = predict_context(model_dict[model_name], word, contexts, image)
    predicted_index = scores.argmax().item()

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "model_name": model_name,
            "target_word": word,
            "contexts": ", ".join(contexts),
            "predicted_context": contexts[predicted_index],
            "predicted_index": predicted_index
        }
    }

    return construct_response(request, response)


@app.post("/models/{model_name}/predict_images")
async def _predict_images(request: Request, model_name: str, payload: PredictImagesPayload = Depends(checker_images), images: List[UploadFile] = File(...)):
    
    if model_name not in model_dict:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail="Model not found")
    
    if len(images)>10 or len(images)<2:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail= "You should send a number of images between 1 and 10")
    
    word = payload.target_word
    context = payload.context

    processed_images = []
    for image_file in images:

        image_content = await image_file.read()
        image_pil = Image.open(BytesIO(image_content))
        processed_image = preproc(image_pil)


        assert processed_image.ndim == 3 and processed_image.shape[0] == 3, "L'immagine preprocessata non ha la forma corretta."
        processed_images.append(processed_image)


    images_tensor = torch.stack(processed_images).unsqueeze(0)  

    scores = predict(model_dict[model_name], [word], [context], images_tensor)
    best_scores, best_indices = torch.max(scores, dim=1)

    response = {
        "message": f"Success - Best scores and indices: {list(zip(best_scores.tolist(), best_indices.tolist()))}",
        "status-code": 200
    }
    return construct_response(request, response)
