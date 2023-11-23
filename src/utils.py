""" Module containing the utils """

import os
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


class VWSDDataset(torch.utils.data.Dataset):

    """ This class describes the Visual Words Senses Disambiguator Dataset """

    # img_dir must contain images ALREADY preprocessed (converted to tensors)
    def __init__(self, img_dir, train_f, gold_f, device='cpu'):
        self.img_dir = img_dir
        self.device = device
        self.word = []
        self.context = []
        self.images = []
        self.true = []
        with open(train_f, encoding='UTF-8') as file_1, open(gold_f, encoding='UTF-8') as file_2:
            for line_1, line_2 in zip(file_1, file_2):
                entry = line_1.split('\t')
                imgs = [x.strip() for x in entry[2:]]
                pos = imgs.index(line_2.strip())
                self.word.append(entry[0])
                self.context.append(entry[1])
                self.images.append(imgs)
                self.true.append(pos)

    def __len__(self):
        return len(self.word)

    def __getitem__(self, i):
        if isinstance(i, slice):
            images = []
            for j in range(0 if i.start is None else i.start, i.stop):
                images.append(torch.stack([torch.load(os.path.join(self.img_dir, img))
                              for img in self.images[j]], dim=0))
            images = torch.cat(images)
        else:
            images = []
            for img in self.images[i]:
                images.append(torch.load(os.path.join(self.img_dir, img)))

            # images = torch.stack(images, dim=0).to(self.device) if self.cache else images
            images = torch.stack(images, dim=0).to(self.device)
        return self.word[i], self.context[i], images, self.true[i]


class Disambiguator():

    """ This class describes the Disambiguator """

    def __init__(self, device='cpu'):
        self.model = SentenceTransformer('all-mpnet-base-v2', device=device)

    def _get_synsets(self, word):
        variants = [word,
                    word.replace('-', '_'),
                    word.replace("'", " ").strip(),
                    ' '.join([x for x in word.split(' ')
                              if x not in stopwords.words('english')]).strip(),
                    ' '.join([x for x in word.replace("'", " ").split(' ')
                              if x not in stopwords.words('english')]).strip()]
        for word_iter in variants:
            synsets = wn.synsets(word_iter.replace(' ', '_'))
            if len(synsets) > 0:
                return synsets
        return synsets

    def get_synsets(self, word):

        """ Method used to get synsets """

        return self._get_synsets(word)

    def _remove_word_from_context(self, word, context):
        return context.replace(word, '').strip()

    def remove_word_from_context(self, word, context):

        """ Method used to remove word from context """

        return self._remove_word_from_context(word, context)

    def mpnet(self, word, context):

        """ Method used for the mpnet """

        senses = []
        for word_iter, context_iter in zip(word, context):
            w_def_all = [x.definition() for x in self._get_synsets(word_iter)]
            c_def_all = [x.definition()
                         for x in self._get_synsets(self._remove_word_from_context(word_iter, context_iter))]
            if len(w_def_all) == 1:
                senses.append(w_def_all[0])
                continue
            if len(w_def_all) == 0 or len(c_def_all) == 0:
                senses.append(f'intended as {context_iter.replace(word_iter, "").strip()}')
                continue
            w_def_emb = self.model.encode(w_def_all, convert_to_tensor=True)
            c_def_emb = self.model.encode(c_def_all, convert_to_tensor=True)
            sim = util.dot_score(w_def_emb, c_def_emb)
            senses.append(w_def_all[sim.argmax() // len(c_def_all)])
        return senses

    def __call__(self, word, context):
        if isinstance(word, list) is False and isinstance(word, tuple) is False:
            word = [word]
        if isinstance(context, list) is False and isinstance(word, tuple) is False:
            context = [context]
        senses = self.mpnet(word, context)
        if len(senses) == 1:
            return senses[0]
        return senses
