import pytest
import torch
from torch.utils.data import DataLoader

from src.models.conf import config
from src.models.utils import VWSDDataset

images_path = config['TRAIN_IMAGES_PATH']
train_data = config['TRAIN_DATA']
target_images = config['TRAIN_TARGET_IMAGES']


def test_ten_images_per_context():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = VWSDDataset(images_path, train_data, target_images, device=dev)

    for word, context, images, true in DataLoader(data, batch_size=1, shuffle=True):
        assert images.shape[1] == 10


if __name__ == "__main__":
    pytest.main()