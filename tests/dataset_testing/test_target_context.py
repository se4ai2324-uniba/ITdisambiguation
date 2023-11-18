import pytest
import torch
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
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
                contexts.pop(0)
                contexts = ' '.join([x for x in contexts if x.lower() not in stopwords.words('english')]).strip()
                assert len(contexts.split()) == 1



if __name__ == "__main__":
    pytest.main()
