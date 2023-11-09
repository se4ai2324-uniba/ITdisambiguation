""" Module used to do the minimum functionality test """

from urllib import request
from os import remove
import pytest
import torch
import open_clip
from PIL import Image
from src.models.evaluate import predict
from src.conf import config
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG1 = 'mouse1.jpg'
IMG2 = 'mouse2.jpg'

def test_minimum_functionality():

    """ Method used to do the minimum functionality test """

    # Changing the context word should change model's output
    model, _, preproc = open_clip.create_model_and_transforms('RN50', 'openai', device=DEV)
    model.load_state_dict(torch.load(config['MODEL_FILE'], map_location=DEV))

    # Mouse animal image
    request.urlretrieve("https://images.theconversation.com/files/"
        + "265294/original/file-20190322-36283-1me4pb6.jpg", IMG1)
    # Computer image
    request.urlretrieve("https://images.unsplash.com/photo-1587831990711-23ca6441447b", IMG2)

    image1 = preproc(Image.open(IMG1))
    image2 = preproc(Image.open(IMG2))

    images1 = torch.stack([image1, image2, image2, image2,
                           image2, image2, image2, image2, image2, image2], dim=0)
    images2 = torch.stack([image2, image1, image1, image1,
                           image1, image1, image1, image1, image1, image1], dim=0)
    images = torch.stack([images1, images2], dim=0)

    remove(IMG1)
    remove(IMG2)
    print(images.shape)

    words = ['mouse', 'computer']
    contexts = ['mouse animal', 'computer device']

    predicted_scores = predict(model, words, contexts, images)
    predicted_images = predicted_scores.argmax(dim=1).tolist()

    assert predicted_images[0] == 0
    assert predicted_images[1] == 0

if __name__ == '__main__':
    pytest.main()
