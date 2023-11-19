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
        if not context.strip():  # Utilizza strip() per gestire anche le stringhe con solo spazi
            raise ValueError("You should send the context")

        word = values.get("target_word", "").strip()  # Gestisci il caso in cui target_word sia None o solo spazi
        if word and word not in context:
            context = f"{word} {context}"

        return context

