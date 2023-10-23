import torch
import open_clip
from torch.utils.data import DataLoader
from ..utils import VWSDDataset, Disambiguator
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_metrics(scores, pos):
    sorted_score = scores.argsort(descending=True)
    ranks = (sorted_score == pos.unsqueeze(0).T).nonzero(as_tuple=True)[1] + 1
    h1 = (ranks == 1).nonzero().flatten().size(0) / scores.size(0)
    h3 = (ranks <= 3).nonzero().flatten().size(0) / scores.size(0)

    return {'mrr': (1 / ranks).mean().item(), 'hits1': h1, 'hits3': h3}

BATCH_SIZE = 32

images_path = '../../data/test_preprocessed_images/'
test_data = '../../data/Test/en.test.data.v1.1.txt'
target_images = '../../data/Test/en.test.gold.v1.1.txt'
model_file = '../../models/model.pt'
output_folder = '../../metrics/'

with torch.no_grad():
    disambiguator = Disambiguator(device=dev)
    tokenizer = open_clip.get_tokenizer('RN50')
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
    model.load_state_dict(torch.load(model_file))
    data = VWSDDataset(images_path, test_data, target_images, device=dev)
    all_scores = torch.empty((0,10))
    all_pos = torch.empty((0,))
    i = 0
    for word, context, images, true in DataLoader(data, batch_size=BATCH_SIZE):
        text = tokenizer([f'This is {c}, {exp}.' for c, exp in zip(context, disambiguator(word, context))]).to(dev)
        text_emb = model.encode_text(text, normalize=True)
        imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

        scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb, imgs_emb.view(text_emb.size(0), 10, -1))).softmax(-1)
        all_scores = torch.cat((all_scores, scores.cpu()))
        all_pos = torch.cat((all_pos, true))
        i += scores.size(0)
        print(f'{i}/{len(data)}')

    results = compute_metrics(all_scores, all_pos)
    for i in results.keys():
        with open(f"{output_folder}{i}.metric", 'w') as f:
            f.write(f"{i.upper()}: {results[i]}\n")
