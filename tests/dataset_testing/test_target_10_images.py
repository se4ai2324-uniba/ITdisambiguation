""" MOdule used to test the target 10 images """

import pytest
from src.conf import config

train_images_names = config["TRAIN_DATA"]


def test_target_10_images():

    """ Method used to test the target 10 images """

    with open(train_images_names, 'r', encoding='UTF-8') as file_r:
        for line in file_r:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                image_filenames = parts[2:]
                assert len(image_filenames) == 10



if __name__ == "__main__":
    pytest.main()
