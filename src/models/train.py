import torch
import open_clip
from conf import config
from utils import VWSDDataset, Disambiguator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import NLLLoss

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
loss = NLLLoss()
model.train()
for epoch in range(EPOCHS):
	i = 0
	for word, context, images, true in DataLoader(data, batch_size=BATCH_SIZE, shuffle=True):
		text = tokenizer([f'This is {c}, {exp}.' for c, exp in zip(context, disambiguator(word, context))]).to(dev)
		text_emb = model.encode_text(text, normalize=True)
		imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

		scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb, imgs_emb.view(text_emb.size(0), 10, -1))).log_softmax(-1)
		l = loss(scores, true.to(dev)) / GRAD_ACC
		l.backward()
		i += scores.size(0)
		print(f'[epoch {epoch} | {i}/{len(data)}] loss: {l}')
		if i % (BATCH_SIZE * GRAD_ACC) == 0 or scores.size(0) < BATCH_SIZE:
			print('Optimization step')
			opt.step()
			opt.zero_grad()
print('Training finished')
torch.save(model.state_dict(), model_file)


#Function for training the model
def train_model(model, data, tokenizer, disambiguator, opt, loss_fn, device, epochs, batch_size, grad_acc):
    """
    Train the model for a given number of epochs.

    Args:
    - model: The model to train.
    - word, context, images, true: Direct batch inputs.
    - tokenizer: The tokenizer for text processing.
    - disambiguator: The disambiguator for word disambiguation.
    - opt: The optimizer.
    - loss_fn: The loss function.
    - device: The training device ('cuda' or 'cpu').
    - num_epochs: Number of epochs to train on the single batch.
    - grad_acc: Gradient accumulation steps.

    Returns:
    - loss_history: A list containing the loss for each epoch.
    - final_lr: Final Learning rate value.
    """
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        i = 0
        for word, context, images, true in DataLoader(data, batch_size=batch_size, shuffle=True):
            text = tokenizer([f'This is {c}, {exp}.' for c, exp in zip(context, disambiguator(word, context))]).to(device)
            text_emb = model.encode_text(text, normalize=True)
            imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

            scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb, imgs_emb.view(text_emb.size(0), 10, -1))).log_softmax(-1)
            l = loss_fn(scores, true.to(device)) / grad_acc
            l.backward()
            epoch_loss += l.item()
            i += scores.size(0)

            if i % (batch_size * grad_acc) == 0 or scores.size(0) < batch_size:
                opt.step()
                opt.zero_grad()

        avg_epoch_loss = epoch_loss / len(data)
        loss_history.append(avg_epoch_loss)

    print('Training finished')
    final_lr = opt.param_groups[0]['lr']  
    return loss_history, final_lr



def train_on_single_batch(model, word, context, images, true, tokenizer, disambiguator, opt, loss_fn, device, num_epochs, grad_acc):
    """
    Train the model on a single batch for a given number of epochs.

    Args:
    - model: The model to train.
    - word, context, images, true: Direct batch inputs.
    - tokenizer: The tokenizer for text processing.
    - disambiguator: The disambiguator for word disambiguation.
    - opt: The optimizer.
    - loss_fn: The loss function.
    - device: The training device ('cuda' or 'cpu').
    - num_epochs: Number of epochs to train on the single batch.
    - grad_acc: Gradient accumulation steps.

    Returns:
    - loss_history: A list containing the loss for each epoch.
    - accuracy_history: A list containing the accuracy for each epoch.
    """
    
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        text = tokenizer([f'This is {c}, {exp}.' for c, exp in zip(context, disambiguator(word, context))]).to(device)
        text_emb = model.encode_text(text, normalize=True)
        imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

        scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb, imgs_emb.view(text_emb.size(0), 10, -1))).log_softmax(-1)
        l = loss_fn(scores, true.to(device))
        l.backward()

        # Gradient accumulation logic
        if (epoch + 1) % grad_acc == 0 or epoch == num_epochs - 1:
            opt.step()
            opt.zero_grad()

        # Save the loss for this epoch
        loss_history.append(l.item())

        # Compute the accuracy for this epoch
        predicted = torch.argmax(scores, dim=1)
        correct = (predicted == true).sum().item()
        accuracy = correct / len(true)
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history

