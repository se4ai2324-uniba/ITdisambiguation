import pytest
import torch
import os
from src.models.conf import config

train_images_names = config["TRAIN_DATA"] 


def test_target_10_images():
    with open(train_images_names, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                label = parts[0]
                image_filenames = parts[2:]
                assert len(image_filenames) == 10



if __name__ == "__main__":
    pytest.main()