""" Module used to test the target context """

import pytest
from src.conf import config

train_images_names = config["TRAIN_DATA"]

def test_target_context():

    """ Method used to test the target context """

    with open(train_images_names, 'r', encoding='UTF-8') as file_r:
        for line in file_r:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                label = parts[0]
                contexts = parts[1].split(' ')
                assert len(contexts) == 2
                assert label == contexts[0]



if __name__ == "__main__":
    pytest.main()
