import torch
import open_clip
from utils import Disambiguator
from conf import config
from fastapi import FastAPI
from typing import Optional

dev = "cuda" if torch.cuda.is_available() else "cpu"
__pretrain_models = {"RN50": "openai",
                     "ViT-B-16": "laion2b_s34b_b88k"}

model_dict = {}
preproc = None
disambiguator: Optional[Disambiguator] = None

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
    # Load disambiguator
    disambiguator = Disambiguator(dev)
