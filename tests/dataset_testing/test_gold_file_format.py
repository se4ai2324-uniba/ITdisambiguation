import pytest
import torch
import os
from src.conf import config

images_path = config["TRAIN_ORIGINAL_IMAGE_PATH"]
def test_gold_file_format():
    files = os.listdir(images_path)

    for file in files:
        assert file.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()