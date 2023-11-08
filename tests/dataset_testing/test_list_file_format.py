import pytest
import torch
import os
from src.conf import config

train_images_names = config["TRAIN_DATA"] 

def test_file_format():
    with open(train_images_names, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                image_filenames = parts[2:]
                for image_filename in image_filenames:
                    assert image_filename.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()