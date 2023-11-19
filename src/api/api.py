import sys
sys.path.append("src")
import torch
import open_clip
from datetime import datetime
from PIL import Image
from conf import config
from models.evaluate import predict_context
from api.schemas import PredictContextPayload
from http import HTTPStatus
from pydantic import ValidationError
from fastapi import FastAPI, HTTPException, status, Request, File, UploadFile, Depends, Form
from fastapi.encoders import jsonable_encoder

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

def checker(data: str = Form(...)):
    try:
        return PredictContextPayload.model_validate_json(data)
    except ValidationError as e:
        raise HTTPException(
            detail=jsonable_encoder(e.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
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
def _predict_context(request: Request, model_name: str, payload: PredictContextPayload = Depends(checker), file: UploadFile = File(...)):
    if model_name in model_dict:
        word = payload.target_word
        contexts = payload.contexts
        image = preproc(Image.open(file.file))

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
    else:
        response = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST
        }

    return construct_response(request, response)
