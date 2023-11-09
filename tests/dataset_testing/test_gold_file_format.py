""" Module used to test the gold file format """

import os
import pytest
from src.conf import config

images_path = config["TRAIN_ORIGINAL_IMAGE_PATH"]
def test_gold_file_format():

    """ Method used to test the gold file format """

    files = os.listdir(images_path)

    for file in files:
        assert file.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()
