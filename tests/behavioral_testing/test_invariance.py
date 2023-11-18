import pytest
import torch
import open_clip
from src.models.evaluate import predict
from src.utils import VWSDDataset
from src.conf import config
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_invariance():
    # Changing the context word should change model's output
    model, _, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
    model.load_state_dict(torch.load(config['MODEL_FILE'], map_location=dev))
    data = VWSDDataset(config['TEST_IMAGES_PATH'], config['TEST_DATA'], config['TEST_TARGET_IMAGES'], device=dev)

    # Data entry relative to the word 'neptune'
    _, _, images, true = data[5]
    # Same word
    words = ['neptune', 'neptune']
    # Different context words but with the same meaning
    contexts = ['neptune statue', 'neptune sculpture']
    images = images.unsqueeze(0).repeat(2,1,1,1,1)

    predicted_scores = predict(model, words, contexts, images)
    predicted_images = predicted_scores.argmax(dim=1).tolist()

    # Same prediction outcomes
    assert predicted_images[0] == predicted_images[1]
    assert predicted_images[0] == true

if __name__ == '__main__':
    pytest.main()