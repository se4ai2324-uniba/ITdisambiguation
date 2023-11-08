import pytest
import torch
import os
from src.conf import config

train_images_names = config["TRAIN_DATA"] 

def test_target_context():
    with open(train_images_names, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                label = parts[0]
                contexts = parts[1].split(' ')
                assert len(contexts) == 2
                assert label == contexts[0]



if __name__ == "__main__":
    pytest.main()