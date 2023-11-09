""" Module used to test the files presence """

import os
import pytest
from src.conf import config

train_images_names = config["TRAIN_DATA"]
images_path = config["TRAIN_ORIGINAL_IMAGE_PATH"]

def test_files_presence():

    """ Method used to test the files presence """

    with open(train_images_names, 'r', encoding='UTF-8') as file_r:
        for line in file_r:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                image_filenames = parts[2:]
                for image_filename in image_filenames:
                    image_path = os.path.join(images_path, image_filename)
                    assert os.path.exists(image_path)


if __name__ == "__main__":
    pytest.main()
