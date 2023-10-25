import dagshub
import mlflow
import torch
import open_clip
from torch.utils.data import DataLoader
from conf import config
from utils import VWSDDataset, Disambiguator
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_metrics(scores, pos):
    sorted_score = scores.argsort(descending=True)
    ranks = (sorted_score == pos.unsqueeze(0).T).nonzero(as_tuple=True)[1] + 1
    h1 = (ranks == 1).nonzero().flatten().size(0) / scores.size(0)
    h3 = (ranks <= 3).nonzero().flatten().size(0) / scores.size(0)

    return {'mrr': (1 / ranks).mean().item(), 'hits1': h1, 'hits3': h3}

BATCH_SIZE = 32

images_path = config['TEST_IMAGES_PATH']
test_data = config['TEST_DATA']
target_images = config['TEST_TARGET_IMAGES']
model_file = config['MODEL_FILE']
output_folder = config['METRICS_FOLDER']

with torch.no_grad():
    disambiguator = Disambiguator(device=dev)
    tokenizer = open_clip.get_tokenizer('RN50')
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', 'openai', device=dev)
    model.load_state_dict(torch.load(model_file, map_location=dev))
    data = VWSDDataset(images_path, test_data, target_images, device=dev)
    all_scores = torch.empty((0,10))
    all_pos = torch.empty((0,))
    i = 0
    print("[+] Starting evaluation [+]")
    for word, context, images, true in DataLoader(data, batch_size=BATCH_SIZE):
        text = tokenizer([f'This is {c}, {exp}.' for c, exp in zip(context, disambiguator(word, context))]).to(dev)
        text_emb = model.encode_text(text, normalize=True)
        imgs_emb = model.encode_image(images.flatten(end_dim=1), normalize=True)

        scores = (100.0 * torch.einsum('ij,ikj->ik', text_emb, imgs_emb.view(text_emb.size(0), 10, -1))).softmax(-1)
        all_scores = torch.cat((all_scores, scores.cpu()))
        all_pos = torch.cat((all_pos, true))
        i += scores.size(0)
        print(f'{i}/{len(data)}')
    print("[+] Finished evaluation [+]")

    results = compute_metrics(all_scores, all_pos)
    for i in results.keys():
        with open(f"{output_folder}{i}.metric", 'w') as f:
            f.write(f"{i.upper()}: {results[i]}\n")

    dagshub.init("ITdisambiguation", "se4ai2324-uniba", mlflow=True)
    mlflow.start_run()
    # Log model file 
    mlflow.log_artifact(model_file)
    mlflow.pytorch.log_model(model, "model")
    # Log model's parameters
    mlflow.log_params({
        'batch_size': config['BATCH_SIZE'],
        'grad_acc_factor': config['GRAD_ACC'],
        'epochs': config['EPOCHS'],
        'learning_rate': config['LEARNING_RATE']
    })
    # Log results
    mlflow.log_metrics(results)
    mlflow.end_run()
