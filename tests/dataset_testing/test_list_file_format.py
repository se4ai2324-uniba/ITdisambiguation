""" Module used to test the list file format """

import pytest
from src.conf import config

train_images_names = config["TRAIN_DATA"]

def test_file_format():

    """ Method used to test the list file format """

    with open(train_images_names, 'r', encoding='UTF-8') as file_r:
        for line in file_r:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                image_filenames = parts[2:]
                for image_filename in image_filenames:
                    assert image_filename.endswith('.jpg')


if __name__ == "__main__":
    pytest.main()
