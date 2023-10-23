import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentence_transformers.util as util
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

class VWSDDataset(torch.utils.data.Dataset):
    # img_dir must contain images ALREADY preprocessed (converted to tensors)
	def __init__(self, img_dir, train_file, gold_file, device='cpu'):
		self.img_dir = img_dir
		self.device = device
		self.word = []
		self.context = []
		self.images = []
		self.true = []
		with open(train_file) as f1, open(gold_file) as f2:
			for l1, l2 in zip(f1, f2):
				entry = l1.split('\t')
				imgs = [x.strip() for x in entry[2:]]
				pos = imgs.index(l2.strip())
				self.word.append(entry[0])
				self.context.append(entry[1])
				self.images.append(imgs)
				self.true.append(pos)
	def __len__(self):
		return len(self.word)
	def __getitem__(self, i):
		if type(i) is slice:
			images = []
			for j in range(0 if i.start is None else i.start, i.stop):
				images.append(torch.stack([torch.load(os.path.join(self.img_dir, img)) for img in self.images[j]], dim=0))
			images = torch.cat(images)
		else:
			images = []
			for img in self.images[i]:
				images.append(torch.load(os.path.join(self.img_dir, img)))
			#images = torch.stack(images, dim=0).to(self.device) if self.cache else images
			images = torch.stack(images, dim=0).to(self.device)
		return self.word[i], self.context[i], images, self.true[i]

class Disambiguator():
	def __init__(self, device='cpu'):
		self.model = SentenceTransformer('all-mpnet-base-v2', device=device)
	def _text_preproc(self, text, tokenize=True):
		stemmer = PorterStemmer()
		tokens = word_tokenize(text) if tokenize else text
		bow = set([stemmer.stem(t.lower()) for t in tokens if t not in stopwords.words('english') and t.isalpha()])
		return bow
	def _get_synsets(self, word):
		variants = [word,
					word.replace('-', '_'),
					word.replace("'", " ").strip(), 
					' '.join([x for x in word.split(' ') if x not in stopwords.words('english')]).strip(),
					' '.join([x for x in word.replace("'", " ").split(' ') if x not in stopwords.words('english')]).strip()]
		for w in variants:
			ss = wn.synsets(w.replace(' ', '_'))
			if len(ss) > 0:
				return ss
		return ss
	def mpnet(self, word, context):
		senses = []
		for w, c in zip(word, context):
			w_def_all = [x.definition() for x in self._get_synsets(w)]
			c_def_all = [x.definition() for x in self._get_synsets(c.replace(w, '').strip())]
			if len(w_def_all) == 1:
				senses.append(w_def_all[0])
				continue
			if len(w_def_all) == 0 or len(c_def_all) == 0:
				senses.append(f'intended as {c.replace(w, "").strip()}')
				continue
			w_def_emb = self.model.encode(w_def_all, convert_to_tensor=True)
			c_def_emb = self.model.encode(c_def_all, convert_to_tensor=True)
			sim = util.dot_score(w_def_emb, c_def_emb)
			senses.append(w_def_all[sim.argmax() // len(c_def_all)])
		return senses
	def __call__(self, word, context):
		if type(word) is not list and type(word) is not tuple:
			word = [word]
		if type(context) is not list and type(word) is not tuple:
			context = [context]
		senses = self.mpnet(word, context)
		if len(senses) == 1:
			return senses[0]
		return senses
