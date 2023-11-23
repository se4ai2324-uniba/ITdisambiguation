import pytest
from src.api.request import send_images_to_api

def test_success():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg"]
    target_word = "mouse"
    context = "pc"

    # Invia la richiesta all'API
    response = send_images_to_api(api_url, image_urls, target_word, context)

    # Controlla lo status code
    assert response.status_code == 200, "Wrong status code"

def test_empty_context():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg"]
    target_word = "mouse"
    context = ""

    response = send_images_to_api(api_url, image_urls, target_word, context)

    assert response.status_code == 422, "Wrong status code"

def test_empty_target_word():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg"]
    target_word = ""
    context = "pc"

    response = send_images_to_api(api_url, image_urls, target_word, context)
    assert response.status_code == 422, "Wrong status code"
    

def test_malformed_context():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = ["https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg"]
    target_word = "mouse"
    context = "pc computer"

    response = send_images_to_api(api_url, image_urls, target_word, context)
    assert response.status_code == 422, "Wrong status code"

def test_more_than_10_images():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = [
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg",
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg",
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg",
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg",
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg", 
                "https://gdsit.cdn-immedia.net/2016/11/TOPOLINO-970x485.jpg",                
                ]
    target_word = "mouse"
    context = "pc computer"

    response = send_images_to_api(api_url, image_urls, target_word, context)
    assert response.status_code ==422, "Wrong status code"

def test_less_than_2_images():
    api_url = "http://localhost:8000/models/RN50/predict_images"
    image_urls = [
                "https://www.magiacomputers.it/media/k2/items/cache/f710044bf79a4b1f5d8b085e5e5d9711_M.jpg"              
                ]
    target_word = "mouse"
    context = "pc computer"

    response = send_images_to_api(api_url, image_urls, target_word, context)
    assert response.status_code ==422, "Wrong status code"

if __name__ == '__main__':
    pytest.main()
