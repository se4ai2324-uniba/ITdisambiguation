import sys

from urllib3 import response
sys.path.append("src")
import torch
import open_clip
from PIL import Image
from conf import config
from models.evaluate import predict_context
from schemas import PredictContextPayload
from http import HTTPStatus
from fastapi import FastAPI, Request

dev = "cuda" if torch.cuda.is_available() else "cpu"
__pretrain_models = {"RN50": "openai",
                     "ViT-B-16": "laion2b_s34b_b88k"}

model_dict = {}
preproc = None

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
    # Load image transformation function
    preproc = open_clip.image_transform(224, False)

@app.post("/models/{model_name}/predict_context")
def _predict_context(request: Request, model_name: str, payload: PredictContextPayload):
    if model_name in model_dict:
        word = payload.target_word
        # Preparing contexts
        contexts = payload.contexts.split(",")
        if len(contexts) < 2:
            return { 
                "message": "Invalid context, make sure to send at least two different contexts",
                "status-code": HTTPStatus.BAD_REQUEST
            }
        for i in range(len(contexts)):
            contexts[i] = contexts[i].strip()
            if word not in contexts[i]:
                contexts[i] = f"{word} {contexts[i]}"

        image = preproc(Image.open(payload.image))

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

    return response
