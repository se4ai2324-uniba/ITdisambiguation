""" Module used to do the directional test """

import pytest
import torch
import open_clip
from src.models.evaluate import predict
from src.utils import VWSDDataset
from src.conf import config
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_directional():

    """ Method used to do the directional test """

    # Changing the context word should change model's output
    model, _, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=DEV)
    model.load_state_dict(torch.load(config['MODEL_FILE'], map_location=DEV))
    data = VWSDDataset(config['TEST_IMAGES_PATH'],
                       config['TEST_DATA'], config['TEST_TARGET_IMAGES'], device=DEV)

    # Data entry relative to the word 'neptune'
    _, _, images, _ = data[5]
    # Same word
    words = ['neptune', 'neptune']
    # Different contexts
    contexts = ['neptune statue', 'neptune planet']
    images = images.unsqueeze(0).repeat(2,1,1,1,1)

    predicted_scores = predict(model, words, contexts, images)
    predicted_images = predicted_scores.argmax(dim=1).tolist()

    # Different prediction outcomes
    assert predicted_images[0] == 6
    assert predicted_images[1] == 5

if __name__ == '__main__':
    pytest.main()
