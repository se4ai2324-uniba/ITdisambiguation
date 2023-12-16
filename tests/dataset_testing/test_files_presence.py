import pytest
import torch
import os
from src.conf import config

train_images_names = config["TRAIN_DATA"] 
images_path = config["TRAIN_IMAGES_PATH"]

def test_files_presence():
    with open(train_images_names, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                label = parts[0]
                image_filenames = parts[2:]
                for image_filename in image_filenames:
                    image_path = os.path.join(images_path, image_filename)
                    assert os.path.exists(image_path)


if __name__ == "__main__":
    pytest.main()
