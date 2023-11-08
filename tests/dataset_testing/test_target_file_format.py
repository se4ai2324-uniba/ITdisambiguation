import pytest
import torch
import os
from src.models.conf import config

target_images_names = config["TRAIN_TARGET_IMAGES"] 

def test_target_file_format():
    with open(target_images_names, 'r') as file:
        image_filenames = file.read().splitlines()

    for filename in image_filenames:
        assert filename.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()