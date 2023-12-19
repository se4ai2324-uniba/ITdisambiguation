import sys

sys.path.append("src")
import re
import torch
import open_clip
from PIL import Image
from conf import config
from models.evaluate import predict_context, predict
from api.schemas import *
from pydantic import ValidationError
from fastapi import FastAPI, HTTPException, status, Request, File, UploadFile, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from typing import List
from io import BytesIO
from api.prometheus.instrumentator import instrumentator
from prometheus_client import start_http_server


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173/"
]

model_dict = {}


@asynccontextmanager
# def _load_models_and_transformation():
async def lifespan(app: FastAPI):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    __pretrain_models = {"RN50": "openai",
                         "ViT-B-16": "laion2b_s34b_b88k"}
    app.state.preproc = open_clip.image_transform(224, False)
    # Load models
    for model_name in __pretrain_models:
        model = open_clip.create_model(model_name,
                                       __pretrain_models[model_name],
                                       dev)
        if model_name == "RN50":
            model.load_state_dict(torch.load(config["MODEL_FILE"],
                                             map_location=dev))
        model_dict[model_name] = model
    yield


app = FastAPI(
    title="Image Text disambiguation APIs",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

def checker_context(target_word: str = Form(...), contexts: str = Form(...)):
    try:
        return PredictContextPayload(target_word=target_word, contexts=contexts)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=jsonable_encoder(e.errors())
        )


def checker_images(target_word: str = Form(...), context: str = Form(...)):
    try:
        return PredictImagesPayload(target_word=target_word, context=context)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=jsonable_encoder(e.errors())
        )


def value_from_metric_content(metric_content: str):
    return float(re.search(r'\d+\.\d+', metric_content).group())


@app.get("/models",
         tags=["Infos"],
         summary="Gets a list of all the available models",
         response_model=GetModelNamesResponseModel)
def _get_models(request: Request):
    model_names = list(model_dict.keys())
    response = GetModelNamesResponseModel(
        data=GetModelNamesData(
            model_names=model_names
        )
    )
    return response


@app.get("/models/{model_name}",
         tags=["Infos"],
         summary="Gets all the infos about the given model",
         response_model=GetModelInfosResponseModel)
def _get_model_infos(request: Request, model_name: str):
    if model_name not in model_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not found")

    model = model_dict[model_name]
    if model_name == "RN50":
        description = (
            "This CLIP model uses ResNet-50 as the visual backbone and has undergone fine-tuning specifically for the Visual Word Sense "
            "Disambiguation (VWSD) task. During fine-tuning, only certain parts of the model are trained to adapt the pre-trained CLIP to the "
            "specific requirements of the task, leveraging the visual feature extraction capabilities of ResNet-50."
        )
        typical_usage = "Ideal when you want a CLIP model with a solid base like ResNet-50, tailored for a specific word disambiguation task based on images."

        with open("metrics/mrr.metric", 'r') as file:
            mrr = value_from_metric_content(file.read())
        with open("metrics/hits1.metric", 'r') as file:
            hits1 = value_from_metric_content(file.read())
        with open("metrics/hits3.metric", 'r') as file:
            hits3 = value_from_metric_content(file.read())
        metrics = ModelMetrics(
            mrr=mrr,
            hits1=hits1,
            hits3=hits1,
        )

    elif model_name == "ViT-B-16":
        description = (
            "This CLIP configuration uses a Vision Transformer with a 16x16 patch size as the visual backbone. The model structure "
            "harnesses the power of transformers for image processing. The model has been pre-trained and can be used for zero-shot "
            "tasks or fine-tuning on specific tasks."
        )
        typical_usage = (
            "Recommended when you want to benefit from the transformer architecture in visual processing. ViT-B-16 is lighter than larger models "
            "but can still provide good performance in computer vision tasks and image-based word disambiguation."
        )
        metrics = ModelMetrics(
            mrr=0.810,
            hits1=0.708,
            hits3=0.894,
        )

    response = GetModelInfosResponseModel(
        data=GetModelInfosData(
            model_name=model_name,
            n_parameters=sum(p.numel() for p in model.parameters()),
            description=description,
            typical_usage=typical_usage,
            metrics=metrics
        )
    )

    return response


@app.post("/models/{model_name}/predict_context",
          tags=["Predictions"],
          summary="Predict the most relevant context given a list of contexts, an image and a target word",
          response_description="The most relevant context and its associated index",
          response_model=PredictContextResponseModel)
def _predict_context(request: Request, model_name: str, payload: PredictContextPayload = Depends(checker_context),
                     image: UploadFile = File(...)):
    """
    Predict Context API for a specific model.

    This endpoint receives a list of contexts, an image, and a target word.
    After processing the inputs with the model, the endpoint returns the most probable
    context in relation to the image and the target word.

    - **image**: An image used by the model to capture the right context.
    - **context**: A list of candidate contexts. The model will select the most
    relevant given the image and the target word.
    - **target_word**: The target word of interest. The model will try to identify the
    most relevant context that corresponds to this word in the specified context.

    The endpoint returns the index of the context with the highest score and
    the context itself.
    """

    if model_name not in model_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not found")

    word = payload.target_word
    contexts = payload.contexts
    image = app.state.preproc(Image.open(image.file))

    scores = predict_context(model_dict[model_name], word, contexts, image)
    predicted_index = scores.argmax().item()

    response = PredictContextResponseModel(
        data=PredictContextResponseData(
            model_name=model_name,
            target_word=word,
            contexts=", ".join(contexts),
            predicted_context=contexts[predicted_index],
            predicted_score=scores.squeeze()[predicted_index].item(),
            predicted_context_index=predicted_index
        )
    )

    return response


@app.post("/models/{model_name}/predict_images",
          tags=["Predictions"],
          summary="Predict the most relevant image given a list of images, a context and a target word",
          response_description="The index of the image and the scores",
          response_model=PredictImageResponseModel)
async def _predict_images(request: Request, model_name: str, payload: PredictImagesPayload = Depends(checker_images),
                          images: List[UploadFile] = File(...)):
    """
    Predict Images API for a specific model.

    This endpoint receives a collection of images, a context, and a target word.
    After processing the images with the model, the endpoint returns the index and score
    of the image deemed most relevant in relation to the provided target word and context.

    - **images**: A list of uploaded images. Each image will be evaluated by the model
    to determine its relevance to the context and the target word.
    - **context**: The context in which the target word is used.
    This context helps the model to interpret the target word as accurately as possible.
    - **target_word**: The target word of interest. The model will try to identify the
    most relevant image that corresponds to this word in the specified context.

    The endpoint returns the index of the image with the highest score, indicating which image
    has been evaluated as most relevant by the model.
    """

    if model_name not in model_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not found")

    if len(images) > 10 or len(images) < 2:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="You should send a number of images between 1 and 10")

    word = payload.target_word
    context = payload.context

    processed_images = []
    for image_file in images:
        image_content = await image_file.read()
        image_pil = Image.open(BytesIO(image_content))
        processed_image = app.state.preproc(image_pil)

        processed_images.append(processed_image)

    images_tensor = torch.stack(processed_images).unsqueeze(0)
    scores = predict(model_dict[model_name], [word], [context], images_tensor)
    best_scores, best_indices = torch.max(scores, dim=1)

    response = PredictImageResponseModel(
        data=PredictImageResponseData(
            model_name=model_name,
            target_word=word,
            context=context,
            predicted_image_index=best_indices.tolist()[0],
            predicted_score=best_scores.tolist()[0]
        )
    )

    return response