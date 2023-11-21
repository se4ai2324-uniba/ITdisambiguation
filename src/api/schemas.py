from pydantic import BaseModel, validator
from nltk.corpus import stopwords
from http import HTTPStatus

class PredictContextPayload(BaseModel):
    target_word: str
    contexts: str

    @validator("target_word")
    def target_word_nonempty(cls, v):
        if not v.strip():
            raise ValueError("target_word must be a valid string")
        return v

    @validator("contexts")
    def check_contexts(cls, contexts, values):
        word = values["target_word"]
        contexts = contexts.split(",")
        if len(contexts) < 2:
            raise ValueError("You should send at least two contexts")

        for i in range(len(contexts)):
            c = contexts[i].strip()
            if not c:
                raise ValueError("One of the contexts is empty")

            is_valid = [w for w in c.split(" ") if w not in stopwords.words("english") and w != word]
            if len(is_valid) > 1:
                raise ValueError("The context must contain at most one meaningful word")

            if word not in c:
                c = f"{word} {c}"
            contexts[i] = c
        return contexts


class PredictImagesPayload(BaseModel):
    target_word: str
    context: str

    @validator("target_word")
    def target_word_nonempty(cls, v):
        if not v.strip():  # Utilizza strip() per gestire anche le stringhe con solo spazi
            raise ValueError("target_word must be a valid string")
        return v

    @validator("context")
    def check_context(cls, context, values):
        word = values.get("target_word", "").strip()  # Gestisci il caso in cui target_word sia None o solo spazi
        c = context.strip()

        if not c:  # Utilizza strip() per gestire anche le stringhe con solo spazi
            raise ValueError("You should send the context")

        is_valid = [w for w in c.split(" ") if w not in stopwords.words("english") and w != word]
        if len(is_valid) > 1:
            raise ValueError("The context must contain at most one meaningful word")

        if word and word not in context:
            context = f"{word} {context}"

        return context

class PredictImageResponseData(BaseModel):
    model_name: str
    target_word: str
    context: str
    predicted_image_index: int
    predicted_score: float

class PredictImageResponseModel(BaseModel):
    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: PredictImageResponseData

class ModelMetrics(BaseModel):
    mrr: float
    hits1: float
    hits3: float

class GetModelInfosData(BaseModel):
    model_name: str
    n_parameters: int
    description: str
    typical_usage:str
    metrics: ModelMetrics

class GetModelInfosResponseModel(BaseModel):
    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: GetModelInfosData