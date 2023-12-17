from locust import HttpUser, task, between
from urllib.request import urlopen
from random import choice
from io import BytesIO

def get_random_images(n: int = 1):
    url = "https://picsum.photos/224/224"
    images = [BytesIO(urlopen(url).read()) for _ in range(n)]
    if n == 1:
        return images[0]
    return images

class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def model_list_task(self):
        self.client.get("/models")

    @task
    def model_detail_task(self):
        model = choice(["RN50", "ViT-B-16"])
        self.client.get(f"/models/{model}")

    @task(5)
    def predict_context_task(self):
        model = choice(["RN50", "ViT-B-16"])
        image = get_random_images(1)
        self.client.post(
            f"/models/{model}/predict_context",
            files={"image": image},
            data={"target_word": "test_word", "contexts": "context1, context2"}
        )

    @task(5)
    def predict_images_task(self):
        model = choice(["RN50", "ViT-B-16"])
        images = get_random_images(4)
        self.client.post(
            f"/models/{model}/predict_images",
            files=[("images", x) for x in images],
            data={"target_word": "test_word", "context": "context_word"}
        )
