import pytest
import torch
import open_clip
from PIL import Image
from urllib import request
from os import remove
from src.models.evaluate import predict
from src.conf import config
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
img1 = 'neptune_statue.jpg'
img2 = 'neptune_planet.jpg'

def test_invariance():
    # Changing the context word should change model's output
    model, _, preproc = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
    model.load_state_dict(torch.load(config['MODEL_FILE'], map_location=dev))

    # Download images
    request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Trento-statue_on_top_of_the_fountain_of_Neptune-side.jpg/675px-Trento-statue_on_top_of_the_fountain_of_Neptune-side.jpg', img1)
    request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/0/06/Neptune.jpg', img2)

    image1 = preproc(Image.open(img1))
    image2 = preproc(Image.open(img2))
    images_all = torch.stack([image1, image2])
    images = torch.stack([images_all, images_all])

    remove(img1)
    remove(img2)

    # Same word
    words = ['neptune', 'neptune']
    # Different context words but with the same meaning
    contexts = ['neptune statue', 'neptune sculpture']

    predicted_scores = predict(model, words, contexts, images)
    predicted_images = predicted_scores.argmax(dim=1).tolist()

    # Same prediction outcomes
    assert predicted_images[0] == predicted_images[1]
    assert predicted_images[0] == 0

if __name__ == '__main__':
    pytest.main()
