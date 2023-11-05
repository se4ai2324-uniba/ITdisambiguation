import pytest
import torch
from src.models.train import train_model
import open_clip
from torch.optim import AdamW
from torch.nn import NLLLoss
from src.models.conf import config
from src.models.utils import VWSDDataset, Disambiguator

BATCH_SIZE = config['BATCH_SIZE']
GRAD_ACC = config['GRAD_ACC']
EPOCHS = config['EPOCHS']
LEARNING_RATE = config['LEARNING_RATE']

images_path = config['TRAIN_IMAGES_PATH']
train_data = config['TRAIN_DATA']
target_images = config['TRAIN_TARGET_IMAGES']
model_file = config['MODEL_FILE']

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
disambiguator = Disambiguator(device=dev)
tokenizer = open_clip.get_tokenizer('RN50')
model, preprocess, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
data = VWSDDataset(images_path, train_data, target_images, device=dev)
opt = AdamW([{'params': model.text_projection},
             {'params': model.visual.attnpool.parameters()}],
            lr=LEARNING_RATE)
loss_fn = NLLLoss()


# Modify EPOCHS for testing to be just 1 for faster tests
test_epochs = 1

# Assert training on CPU
loss_history_cpu = train_model(model, data, tokenizer, disambiguator, opt, loss_fn, 'cpu', test_epochs, BATCH_SIZE, GRAD_ACC)
assert loss_history_cpu

# If CUDA is available, assert training on CUDA
if torch.cuda.is_available():
    loss_history_cuda = train_model(model, data, tokenizer, disambiguator, opt, loss_fn, 'cuda', test_epochs, BATCH_SIZE, GRAD_ACC)
    assert loss_history_cuda
