from locust import HttpUser, task, between
from urllib.request import urlopen
from random import choice, sample
from io import BytesIO

class MyUser(HttpUser):
    host = "https://itdisambiguation.azurewebsites.net"
    wait_time = between(1, 5)

    def on_start(self):
        self.rand_images = [BytesIO(urlopen("https://picsum.photos/224/224").read()) for _ in range(15)]

    def get_random_images(self, n: int = 1):
        images = sample(self.rand_images, n)
        if n == 1:
            return images[0]
        return images

    @task
    def model_list_task(self):
        self.client.get("/models")

    @task
    def model_detail_task(self):
        model = choice(["RN50", "ViT-B-16"])
        self.client.get(f"/models/{model}")

    @task(3)
    def predict_context_task(self):
        model = choice(["RN50", "ViT-B-16"])
        image = self.get_random_images(1)
        self.client.post(
            f"/models/{model}/predict_context",
            files={"image": image},
            data={"target_word": "test_target", "contexts": "test_context1, test_context2"}
        )
        image.seek(0)

    @task(5)
    def predict_images_task(self):
        model = choice(["RN50", "ViT-B-16"])
        images = self.get_random_images(4)
        self.client.post(
            f"/models/{model}/predict_images",
            files=[("images", x) for x in images],
            data={"target_word": "test_target", "context": "test_context"}
        )
        for i in images:
            i.seek(0)
