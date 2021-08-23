import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from gensim.models import KeyedVectors
import time



class Classifier(nn.Module):


	def __init__(self, weights, maxlen, n_filters, filter_sizes, gru_size, n_layers, batch_size, prob, device, target, heads=10):

		super(Classifier, self).__init__()
		self.batch_size = batch_size
		self.filter_sizes = filter_sizes
		self.gru_size = gru_size
		self.n_filters = n_filters
		self.word_dim = weights.shape[1]
		self.heads = heads

		self.head_dim = self.word_dim//self.heads
		assert (self.word_dim == self.heads*self.head_dim), "The word dimension is divisible by the number of heads!"

		self.embeddings = nn.Embedding.from_pretrained(weights)
		self.dropout = nn.Dropout(prob)

		if target:
			self.queries = nn.Linear(self.head_dim, self.head_dim)
			self.keys = nn.Linear(self.head_dim, self.head_dim)
			self.values = nn.Linear(self.head_dim, self.head_dim)

		self.convs = nn.ModuleList([
		nn.Conv1d(maxlen, n_filters, filter_size) for filter_size in filter_sizes
		])

		conv_size = 0

		for filter_size in filter_sizes:
			post_conv = self.word_dim - filter_size + 1
			conv_size += post_conv//filter_size            

		self.bi_gru = nn.GRU(conv_size, gru_size, bidirectional=True, batch_first=True, num_layers=n_layers)
		self.bi_gru_prime = nn.GRU(1, gru_size, bidirectional=True, batch_first=True, num_layers=n_layers)

		self.attention_fc = nn.Linear(2*gru_size, 2*gru_size)
		self.u_w = torch.rand(size=(2*gru_size, 1), requires_grad=True, device=device)

		self.attention_fc_prime = nn.Linear(2*gru_size, 2*gru_size)
		self.u_s = torch.rand(size=(2*gru_size, 1), requires_grad=True, device=device)

		self.fc = nn.Linear(2*gru_size, 1)


	def forward(self, inputs, masks, targets):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.embeddings(inputs))

		# MultiHead Attention to add the target information
		if target:
			target_embeddings = self.dropout(self.embeddings(targets))

			Q = self.queries(word_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))
			K = self.keys(target_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))
			V = self.values(target_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))

			dot = torch.matmul(Q.reshape(self.batch_size, self.heads, -1, self.head_dim), K.reshape(self.batch_size, self.heads, self.head_dim, -1))

			if masks is not None:
				masks = masks.reshape(self.batch_size, 1, Q.shape[1], 1)
				dot = dot.masked_fill_(masks == 0, float(0))

			div = dot/self.word_dim**(1/2)

			a = F.softmax(div, dim=-1)
			h = torch.matmul(a.reshape(self.batch_size, self.heads, -1, target_embeddings.shape[1]), V.reshape(self.batch_size, self.heads, -1, self.head_dim))
			word_embeddings = h.reshape(self.batch_size, -1, self.heads*self.head_dim)

		tmp = []

		for i, conv in enumerate(self.convs):
			x = conv(word_embeddings)
			x = F.max_pool1d(x, self.filter_sizes[i])
			tmp.append(x)

		conv_out = torch.cat(tmp, dim=-1)

		h, _ = self.bi_gru(conv_out)

		attention_fc = torch.tanh(self.attention_fc(h))
		a = F.softmax(torch.matmul(attention_fc, self.u_w), dim=1)
		s = torch.matmul(h.reshape(-1, 2*self.gru_size, self.n_filters), a)

		h_prime, _ = self.bi_gru_prime(s)
		attention_fc_prime = torch.tanh(self.attention_fc_prime(h_prime))
		a_prime = F.softmax(torch.matmul(attention_fc_prime, self.u_s), dim=-1)
		u = torch.matmul(h_prime.reshape(-1, 2*self.gru_size, 2*self.gru_size), a_prime)
		u = u.squeeze()

		out = self.fc(u)
		return out



def fit(model, train_loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs, masks, targets, aspects in train_loader: 
		inputs = inputs.to(device)
		masks = masks.to(device)
		aspects = aspects.to(device)
		targets = targets.to(device)

		model.zero_grad()
		outputs = model(inputs, masks, targets)
		loss = criterion(outputs, aspects.reshape(-1, 1))
		loss.backward()
		optimizer.step()
		losses.append(loss)
		total += 1

	epoch_loss = sum(losses)/total
	return epoch_loss



def evaluate(model, valid_loader, criterion, device, predict=False):

	model.eval()
	losses = []
	total = 0

	with torch.no_grad():
		if not predict:
			for inputs, masks, targets, aspects in valid_loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				aspects = aspects.to(device)
				targets = targets.to(device)

				outputs = model(inputs, masks, targets)
				loss = criterion(outputs, aspects.reshape(-1, 1))
				losses.append(loss)
				total += 1

			epoch_loss = sum(losses)/total

			return epoch_loss
		else:
			preds = []

			for inputs, masks, targets, aspects in valid_loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				aspects = aspects.to(device)
				targets = targets.to(device)

				outputs = model(inputs, masks, targets)
				outputs = torch.sigmoid(outputs)

				preds += torch.round(outputs.cpu())

			return pad_sequence(preds, batch_first=True)



if __name__ == "__main__":

	# Import the necessary files
	texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = fileLoader('absa_dataset.txt')
	contractions = contractionsLoader('contractions.txt')
	airlines = extraLoader('airlinesNew.txt')
	aircrafts = extraLoader('aircraftsNew.txt')
	misc = extraLoader('miscNew.txt')
	airports = extraLoader('airportsNew.txt')

	# Create the dataframe
	cols = {'texts': texts, 'targets': targets, 'aspects': aspects, 'categories': aspect_cats, 'attributes': aspect_attrs, 'sentiments': sentiments}
	df = pd.DataFrame(cols)
	# Create a list of the entities in each unique sentence
	c = df.groupby('texts', sort=False)['categories'].agg(lambda x: '  '.join(set(x)))
	# Create a list of the attributes in each unique sentence
	a = df.groupby('texts', sort=False)['attributes'].agg(lambda x: '  '.join(set(x)))
	# Create a list of the targets in each unique sentence
	t = df.groupby('texts', sort=False)['targets'].agg(lambda x: '  '.join(set(x)))
	# Create a list of unique sentences
	unique_texts = df.drop_duplicates('texts')['texts'].reset_index(drop=True)


	# Replace the special tokens in the unique sentences
	new_texts = unique_texts.apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))


	# Replace the special tokens in the targets
	new_targets = t.apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))


	# Some preprocessing of the unique texts and targets
	new_texts = new_texts.apply(textPreprocessing, args=(contractions, ))
	new_targets = new_targets.apply(textPreprocessing, args=(contractions, ))


	# Create the vocabulary
	word_to_ix = prepareVocab(new_texts)
	# Create the word and target sequences (for the Category Detection with Target) as well as the attention masks
	word_sequences = toSequences(new_texts, word_to_ix)
	target_sequences = toSequences(new_targets, word_to_ix)
	attention_masks = attentionMasks(new_texts)


	# Set the parameters
	torch.manual_seed(4)
	vocab_size = len(word_to_ix) + 1
	maxlen = word_sequences.shape[1]
	n_filters = 128
	filter_sizes = [3, 4, 5]
	prob = 0.1
	gru_size = 50
	attention_size = 100
	n_layers = 2
	batch_size = 64
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	lr = 0.001
	epochs = 25
	target = True


	# Load the domain-specific word embeddings
	w2v = KeyedVectors.load('word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	weights = embedding_weight.to(device)


	# Define onehot representations of the entities
	entities = c.str.get_dummies(sep=' ').values
	airline = entities[:, 0]
	ambience = entities[:, 1]
	connectivity = entities[:, 2]
	entertainment = entities[:, 3]
	foodANDdrinks = entities[:, 4]
	seat = entities[:, 5]
	service = entities[:, 6]


	# Define onehot representations of the attributes
	attributes = a.str.get_dummies(sep=' ').values
	cabin = attributes[:, 0]
	comfort = attributes[:, 1]
	general = attributes[:, 2]
	ground = attributes[:, 3]
	options = attributes[:, 4]
	prices = attributes[:, 5]
	quality = attributes[:, 6]
	schedule = attributes[:, 7]
	wifi = attributes[:, 8]


	# Train for the "AIRLINE" entity
	cats = torch.FloatTensor(airline)
	weight = compute_class_weight('balanced', classes=np.unique(cats), y=cats.numpy().reshape(-1))
	weight = torch.FloatTensor(weight)
	weight = weight.to(device)


	# Set the model, the loss function, and the optimizer
	model = Classifier(weights, maxlen, n_filters, filter_sizes, gru_size, n_layers, batch_size, prob, device, target)
	model = model.to(device)
	criterion = nn.BCEWithLogitsLoss(weight=weight[1])
	optimizer = optim.Adam(model.parameters(), lr=lr)


	# Create the training and validation data
	input_train, input_val, mask_train, mask_val, target_train, target_val, cat_train, cat_val = train_test_split(word_sequences, attention_masks, target_sequences, cats,\
																												  test_size=0.1, random_state=4) 
	train_dataset = TensorDataset(input_train, mask_train, target_train, cat_train)
	val_dataset = TensorDataset(input_val, mask_val, target_val, cat_val) 
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Train and validate the model
	for epoch in range(epochs):
		start = time.time()

		train_loss = fit(model, train_loader, criterion, optimizer, device)
		test_loss = evaluate(model, val_loader, criterion, device=device)

		finish = time.time()
		secs = timer(start, finish)

		print(f'Epoch {epoch + 1}/{epochs} ==> {secs:.0f}s - loss: {train_loss} - val_loss: {test_loss}')


	# Print the results of the model in the validation data
	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(cat_val.numpy(), preds))
	sns.heatmap(confusion_matrix(cat_val.numpy(), preds), cmap='Blues', annot=True, cbar=False, fmt='.0f')
	plt.show()