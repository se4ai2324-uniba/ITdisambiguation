import requests

url = "http://127.0.0.1:8000/models/RN50/predict_context"
img = open("data/Test/resized_test_images_N/image.4457.jpg", "rb")
files = {"image": img}
data = {
    "data": '{"target_word": "neptune", "contexts": "statue,planet"}'
}
res = requests.post(url, data=data, files=files)

print(res.json())
