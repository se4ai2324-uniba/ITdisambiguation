from pydantic import BaseModel, validator

class PredictContextPayload(BaseModel):
    target_word: str
    contexts: str

    @validator("target_word")
    def target_word_nonempty(cls, v):
        if len(v) == 0:
            raise ValueError("target_word must be a valid string")
        return v

    @validator("contexts")
    def check_contexts(cls, contexts, values):
        word = values["target_word"]
        contexts = contexts.split(",")
        if len(contexts) < 2:
            raise ValueError("You should send at least two contexts")
        for i in range(len(contexts)):
            contexts[i] = contexts[i].strip()
            if word not in contexts[i]:
                contexts[i] = f"{word} {contexts[i]}"
        return contexts

