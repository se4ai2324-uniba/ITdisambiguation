import pytest
import torch
from torch.utils.data import DataLoader

from src.conf import config
from src.utils import VWSDDataset, Disambiguator

images_path = config['TRAIN_IMAGES_PATH']
train_data = config['TRAIN_DATA']
target_images = config['TRAIN_TARGET_IMAGES']


def test_context_length_and_context_not_containing_target():

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    disambiguator = Disambiguator(device=dev)
    data = VWSDDataset(images_path, train_data, target_images, device=dev)

    for word, context, images, true in DataLoader(data, batch_size=1, shuffle=True):
        new_context = disambiguator._remove_word_from_context(word[0], context[0]).split()
        assert len(new_context) == 1
        assert new_context != word[0]


if __name__ == "__main__":
    pytest.main()
