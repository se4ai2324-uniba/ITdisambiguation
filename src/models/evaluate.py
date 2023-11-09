""" Module used to evaluate the model """

import sys
from torch.utils.data import DataLoader
from conf import config
from utils import VWSDDataset, Disambiguator
import dagshub
import mlflow
import torch
import open_clip
sys.path.append('src')

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
disambiguator = Disambiguator(device=DEV)
tokenizer = open_clip.get_tokenizer('RN50')

BATCH_SIZE = 32
images_path = config['TEST_IMAGES_PATH']
test_data = config['TEST_DATA']
target_images = config['TEST_TARGET_IMAGES']
model_file = config['MODEL_FILE']
output_folder = config['METRICS_FOLDER']


def compute_metrics(scores_tot, pos):

    """ Method used to compute metrics """

    sorted_score = scores_tot.argsort(descending=True)
    ranks = (sorted_score == pos.unsqueeze(0).T).nonzero(as_tuple=True)[1] + 1
    hits1 = (ranks == 1).nonzero().flatten().size(0) / scores_tot.size(0)
    hits3 = (ranks <= 3).nonzero().flatten().size(0) / scores_tot.size(0)

    return {'mrr': (1 / ranks).mean().item(), 'hits1': hits1, 'hits3': hits3}

def predict(model_1, words, contexts, images_1):

    """ Method used to make the prediction """

    text = tokenizer([f'This is {c}, {exp}.'
                      for c, exp in zip(contexts, disambiguator(words, contexts))]).to(DEV)
    text_emb = model_1.encode_text(text, normalize=True)
    imgs_emb = model_1.encode_image(images_1.flatten(end_dim=1), normalize=True)
    scores_tot = (100.0 * torch.einsum('ij,ikj->ik', text_emb,
                                   imgs_emb.view(text_emb.size(0), 10, -1))).softmax(-1)
    return scores_tot

if __name__ == '__main__':
    with torch.no_grad():
        model, _, _ = open_clip.create_model_and_transforms('RN50', 'openai', device=DEV)
        model.load_state_dict(torch.load(model_file, map_location=DEV))
        data = VWSDDataset(images_path, test_data, target_images, device=DEV)
        all_scores = torch.empty((0,10))
        all_pos = torch.empty((0,))
        i = 0
        print("[+] Starting evaluation [+]")
        for word, context, images, true in DataLoader(data, batch_size=BATCH_SIZE):
            scores = predict(model, word, context, images)
            all_scores = torch.cat((all_scores, scores.cpu()))
            all_pos = torch.cat((all_pos, true))
            i += scores.size(0)
            print(f'{i}/{len(data)}')
        print("[+] Finished evaluation [+]")

        results = compute_metrics(all_scores, all_pos)
        for k, _v in results.items():
            with open(f"{output_folder}{k}.metric", 'w', encoding='UTF-8') as f:
                f.write(f"{k.upper()}: {_v}\n")

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
