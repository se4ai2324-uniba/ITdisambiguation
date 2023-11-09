""" Module used to train the model """

import torch
import open_clip
from src.conf import config
from src.utils import VWSDDataset, Disambiguator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import NLLLoss

#Function for training the model
def train_model(num_epochs=config['EPOCHS'], batch_size = config['BATCH_SIZE'],
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the model for a given number of epochs.

    Args:
    - num_epochs: Number of epochs to train on a single batch.
    - batch_size: Size of the batch to use.
    - dev: Device to use for training ('cuda' or 'cpu').


    Returns:
    - loss_history: A list containing the loss for each epoch.
    - final_lr: Final learning rate value.
    """
    grad_acc = config['GRAD_ACC']
    learning_rate = config['LEARNING_RATE']

    images_path = config['TRAIN_IMAGES_PATH']
    train_data = config['TRAIN_DATA']
    target_images = config['TRAIN_TARGET_IMAGES']
    model_file = config['MODEL_FILE']


    disambiguator = Disambiguator(device=dev)
    tokenizer = open_clip.get_tokenizer('RN50')
    model, _preprocess, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
    data = VWSDDataset(images_path, train_data, target_images, device=dev)
    opt = AdamW([{'params': model.text_projection},
                 {'params': model.visual.attnpool.parameters()}],
                lr=learning_rate)
    loss = NLLLoss()
    model.train()
    loss_history_1 = []
    epoch_loss = 0
    average_epoch_loss=0
    for epoch in range(num_epochs):
        i = 0
        for word, context, images, true in DataLoader(data, batch_size=batch_size, shuffle=True):
            text = tokenizer([f'This is {c}, {exp}.'
                              for c, exp in zip(context, disambiguator(word, context))]).to(dev)
            text_emb = model.encode_text(text, normalize=True)
            imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

            scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb,
                                           imgs_emb.view(text_emb.size(0), 10, -1))).log_softmax(-1)
            loss_score = loss(scores, true.to(dev)) / grad_acc
            loss_score.backward()
            epoch_loss += loss_score.item() * grad_acc
            i += scores.size(0)
            print(f'[epoch {epoch} | {i}/{len(data)}] loss: {loss_score}')
            if i % (batch_size * grad_acc) == 0 or scores.size(0) < batch_size:
                print('Optimization step')
                opt.step()
                opt.zero_grad()
        average_epoch_loss = epoch_loss / len(data)
        loss_history_1.append(average_epoch_loss)

    print('Training finished')
    final_lr_1 = opt.param_groups[0]['lr']
    torch.save(model.state_dict(), model_file)
    return loss_history_1, final_lr_1



# Function for training the model
def test_overfit_single_batch(num_epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'],
                              dev='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test the model to check if it overfits on a single batch for a given number of epochs.

    Args:
    - num_epochs: Number of epochs to train on a single batch.
    - batch_size: Size of the batch to use.
    - dev: Device to use for training ('cuda' or 'cpu').

    Returns:
    - loss_history: A list containing the loss for each epoch on the single batch.
    """
    grad_acc = config['GRAD_ACC']
    learning_rate = config['LEARNING_RATE']

    images_path = config['TRAIN_IMAGES_PATH']
    train_data = config['TRAIN_DATA']
    target_images = config['TRAIN_TARGET_IMAGES']

    disambiguator = Disambiguator(device=dev)
    tokenizer = open_clip.get_tokenizer('RN50')
    model, _preprocess, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)


    data = VWSDDataset(images_path, train_data, target_images, device=dev)
     # Only use the first batch of the dataset
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    single_batch_data = next(iter(data_loader))

    opt = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = NLLLoss()

    model.train()
    loss_history_2 = []

    for epoch in range(num_epochs):
        opt.zero_grad()

        word, context, images, true = single_batch_data
        text = tokenizer([f'This is {c}, {exp}.'
                          for c, exp in zip(context, disambiguator(word, context))]).to(dev)
        text_emb = model.encode_text(text, normalize=True)
        imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

        scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb,
                                       imgs_emb.view(text_emb.size(0), 10, -1))).log_softmax(-1)
        loss = loss_fn(scores, true.to(dev)) / grad_acc

        loss.backward()
        opt.step()

        loss_history_2.append(loss.item())
        print(f'Epoch {epoch}: Loss {loss.item()}')

    return loss_history_2

if __name__ == "__main__":
    loss_history, final_lr = train_model()
    print(f"Final learning rate: {final_lr}")
    print(f"Loss history over epochs: {loss_history}")
