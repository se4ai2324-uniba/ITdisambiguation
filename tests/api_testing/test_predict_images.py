import pytest
from io import BytesIO
from urllib.request import urlopen
from src.api.api import app
from fastapi import status
from fastapi.testclient import TestClient

# Workaround to trigger startup event
with TestClient(app) as client:
    pass

def send_images_to_api(client, image_urls, target_word, context):
    """Scarica le immagini e le invia all'API."""
    files = []
    for url in image_urls:
        res = urlopen(url)
        files.append(('images', BytesIO(res.read())))

    # Send images to API
    response = client.post(
        "models/RN50/predict_images",
        files=files,
        data={'target_word': target_word, 'context': context}
    )

    for _, file in files:
        file.close()
    
    return response

def test_success():
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg"]
    target_word = "mouse"
    context = "pc"

    # Send request to API
    response = send_images_to_api(client, image_urls, target_word, context)

    # Check status code
    assert response.status_code == status.HTTP_200_OK, "Wrong status code"

@pytest.mark.parametrize(
    "target_word, context",
    [
        ("mouse", ""),              # Empty context
        ("", "pc"),                 # Empty target word
        ("mouse", "pc computer")    # Malformed context
    ]
)
def test_failure(target_word, context):
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg"]
    target_word = "mouse"
    context = ""

    response = send_images_to_api(client, image_urls, target_word, context)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, "Wrong status code"

@pytest.mark.parametrize(
    "image_urls",
    [
        # More then ten images
        ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg",
         "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg",
         "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg",
         "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg",
         "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg",
         "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg",
         "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg",
         "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg",
         "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg",
         "https://assets.gds.it/2016/11/TOPOLINO-970x485.jpg"],
        # Less then two images
        ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg"]
    ]
)
def test_failure_images(image_urls):
    target_word = "mouse"
    context = "pc computer"

    response = send_images_to_api(client, image_urls, target_word, context)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, "Wrong status code"

if __name__ == '__main__':
    pytest.main()
