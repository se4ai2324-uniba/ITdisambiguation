import sys
import torch
import open_clip
from ..utils import VWSDDataset, Disambiguator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import NLLLoss

if len(sys.argv) < 3:
    sys.exit("Usage: python train.py EPOCHS LEARNING_RATE")

BATCH_SIZE = 8
GRAD_ACC = 8
EPOCHS = int(sys.argv[1])           # 2
LEARNING_RATE = float(sys.argv[2])  # 2e-5

images_path = '../../data/train_preprocessed_data/'
train_data = '../../data/Train/resized_train.data.v1.txt'
target_images = '../../data/Train/resized_train.gold.v1.txt'
model_file = '../../models/model.pt'

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
