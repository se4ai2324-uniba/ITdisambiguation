""" Module used to define schemas """

from http import HTTPStatus
from typing import List
from pydantic import BaseModel, validator
from nltk.corpus import stopwords


class PredictContextPayload(BaseModel):

    """ Class used for the prediction of the context payload """

    target_word: str
    contexts: str

    @validator("target_word")
    def target_word_nonempty(self, v):

        """ Method to verify that the target word is not empty """

        if not v.strip():
            raise ValueError("target_word must be a valid string")
        return v

    @validator("contexts")
    def check_contexts(self, contexts, values):

        """ Method to check the contexts """

        word = values["target_word"]
        contexts = contexts.split(",")
        if len(contexts) < 2:
            raise ValueError("You should send at least two contexts")

        for i, c in enumerate(contexts):
            c = c.strip()
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

    """ Class used for the prediction of the images payload """

    target_word: str
    context: str

    @validator("target_word")
    def target_word_nonempty(self, v):

        """ Method to verify that the target word is not empty """

        if not v.strip():  # Utilizza strip() per gestire anche le stringhe con solo spazi
            raise ValueError("target_word must be a valid string")
        return v

    @validator("context")
    def check_context(self, context, values):

        """ Method to check the contexts """

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


class PredictContextResponseData(BaseModel):

    """ Class used for the prediction of the context response data """

    model_name: str
    target_word: str
    contexts: str
    predicted_context: str
    predicted_score: float
    predicted_context_index: int


class PredictContextResponseModel(BaseModel):

    """ Class used for the prediction of the context response model """

    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: PredictContextResponseData


class PredictImageResponseData(BaseModel):

    """ Class used for the prediction of the image response data """

    model_name: str
    target_word: str
    context: str
    predicted_image_index: int
    predicted_score: float


class PredictImageResponseModel(BaseModel):

    """ Class used for the prediction of the image response model """

    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: PredictImageResponseData


class ModelMetrics(BaseModel):

    """ Class that describes the model metrics """

    mrr: float
    hits1: float
    hits3: float


class GetModelInfosData(BaseModel):

    """ Class used to get infos on the model """

    model_name: str
    n_parameters: int
    description: str
    typical_usage: str
    metrics: ModelMetrics


class GetModelInfosResponseModel(BaseModel):

    """ Class used to get infos on the model response """

    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: GetModelInfosData


class GetModelNamesData(BaseModel):

    """ Class used to get the model names data """

    model_names: List[str]


class GetModelNamesResponseModel(BaseModel):

    """ Class used to get the response of model names """

    message: str = HTTPStatus.OK.phrase
    status_code: int = HTTPStatus.OK.value
    data: GetModelNamesData
