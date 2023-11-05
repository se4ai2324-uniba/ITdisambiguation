import pytest
import torch
from src.models.train import train_on_single_batch
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


def test_overfit_batch():
    # Extract a single batch of data.
    word, context, images, true = data[0:BATCH_SIZE]

    # Increase the number of epochs for overfitting
    overfit_epochs = 2

    # Train the model on the single batch
    loss_history, accuracy_history = train_on_single_batch(model, word, context, images, true, tokenizer, disambiguator, opt, loss_fn, dev, overfit_epochs, GRAD_ACC)

    # Extract the final accuracy from the accuracy history
    final_accuracy = accuracy_history[-1]

    # Assert that the final accuracy is greater than 0.95
    assert final_accuracy > 0.95


if __name__ == "__main__":
    pytest.main()

