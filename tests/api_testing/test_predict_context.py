import pytest
from fastapi import status
from src.api.request import send_context_to_api

def test_success():
    image_path = "data/Test/resized_test_images_N/image.91.jpg"
    target_word = "aquila"
    contexts = "bird, constellation"

    response = send_context_to_api(image_path, target_word, contexts)
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
def test_insuccess(target_word, contexts):
    image_path = "data/Test/resized_test_images_N/image.91.jpg"

    response = send_context_to_api(image_path, target_word, contexts)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

if __name__ == "__main__":
    pytest.main()
