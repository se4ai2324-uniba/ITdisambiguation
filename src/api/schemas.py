from io import BufferedReader
from pydantic import BaseModel

class PredictContextPayload(BaseModel):
    target_word: str
    contexts: str
    image: BufferedReader
