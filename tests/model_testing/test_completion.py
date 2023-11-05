import os
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
#EPOCHS = config['EPOCHS']
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


def test_training_completion():
    loss_history, final_lr = train_model(model, data, tokenizer, disambiguator, opt, loss_fn, dev, num_epochs=1, grad_acc=GRAD_ACC)

    # Assert that the final learning rate is greater than a specified minimum
    min_learning_rate = 0.00001
    assert final_lr >= min_learning_rate, f"Final learning rate {final_lr} is less than minimum threshold {min_learning_rate}"

    # Assert that the loss history is not empty, implying training iterations occurred
    assert loss_history['train_loss'], "Train loss history is empty."

    assert os.path.exists(model_file), f"File not found {model_file}"

if __name__ == "__main__":
    pytest.main()