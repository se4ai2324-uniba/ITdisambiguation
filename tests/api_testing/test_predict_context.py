import pytest
from io import BytesIO
from urllib.request import urlopen
from fastapi import status
from fastapi.testclient import TestClient
from src.api.api import app

# Workaround to trigger startup event
with TestClient(app) as client:
    pass

def send_context_to_api(client, image_url, target_word, contexts):
    res = urlopen(image_url)
    image = BytesIO(res.read())

    response = client.post(
        "/models/RN50/predict_context",
        files={"image": image},
        data={"target_word": target_word, "contexts": contexts}
    )

    image.close()
    return response

def test_success():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/AquilaCC.jpg/440px-AquilaCC.jpg"
    target_word = "aquila"
    contexts = "bird, constellation"

    response = send_context_to_api(client, image_url, target_word, contexts)
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.parametrize(
    "target_word, contexts",
    [
        ("", "bird, constellation"),                # Empty target word
        ("aquila", ""),                             # Empty contexts
        ("aquila", "bird"),                         # Only one context
        ("aquila", "bird, constellation stars")     # More than two meaningful words in one context
    ]
)
def test_failure(target_word, contexts):
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/AquilaCC.jpg/440px-AquilaCC.jpg"

    response = send_context_to_api(client, image_url, target_word, contexts)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

if __name__ == "__main__":
    pytest.main()
