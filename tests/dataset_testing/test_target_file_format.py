""" Module used to test target file format """

import pytest
from src.conf import config

target_images_names = config["TRAIN_TARGET_IMAGES"]

def test_target_file_format():

    """ Method used to test the target file format """

    with open(target_images_names, 'r', encoding='UTF-8') as file_r:
        image_filenames = file_r.read().splitlines()

    for filename in image_filenames:
        assert filename.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()
