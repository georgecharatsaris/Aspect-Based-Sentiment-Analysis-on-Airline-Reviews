import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import time
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors



class Classifier(nn.Module):


	def __init__(self, vocab_size, weights, maxlen, num_filters, filter_sizes, batch_size, prob, target, heads=10):

		super(Classifier, self).__init__()
		self.batch_size = batch_size
		self.filter_sizes = filter_sizes
		self.heads = heads
		self.word_dim = weights.shape[1]
		self.head_dim = self.word_dim//self.heads
		assert (self.word_dim == self.heads*self.head_dim), "The word dimension should be divisible by the number of heads!"

		self.embeddings = nn.Embedding.from_pretrained(weights)

		self.convs = nn.ModuleList([
		nn.Conv1d(maxlen, num_filters, filter_size) for filter_size in filter_sizes
		])
		self.dropout = nn.Dropout(prob)

		if target:
			self.queries = nn.Linear(self.head_dim, self.head_dim)
			self.keys = nn.Linear(self.head_dim, self.head_dim)
			self.values = nn.Linear(self.head_dim, self.head_dim)

		self.fc = nn.Linear(len(filter_sizes)*num_filters, 1)


	def forward(self, inputs, masks, aspects):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.embeddings(inputs))

		# MultiHead Attention to add the target information
		if target:
			target_embeddings = self.dropout(self.embeddings(aspects))

			Q = self.queries(word_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))
			K = self.keys(target_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))
			V = self.values(target_embeddings.reshape(self.batch_size, -1, self.heads, self.head_dim))

			dot = torch.matmul(Q.reshape(self.batch_size, self.heads, -1, self.head_dim), K.reshape(self.batch_size, self.heads, self.head_dim, -1))

			if masks is not None:
				masks = masks.reshape(self.batch_size, -1, Q.shape[1], 1)
				dot = dot.masked_fill_(masks == 0, float(0))

			div = dot/self.word_dim**(1/2)

			a = F.softmax(div, dim=-1)
			h = torch.matmul(a.reshape(self.batch_size, self.heads, -1, target_embeddings.shape[1]), V.reshape(self.batch_size, self.heads, -1, self.head_dim))
			word_embeddings = h.reshape(self.batch_size, -1, self.heads*self.head_dim)

		out = []

		for i, conv in enumerate(self.convs):
			x = F.relu(conv(word_embeddings))
			x = F.max_pool1d(x, self.word_dim - self.filter_sizes[i] + 1)
			out.append(x.squeeze(-1))

		out = torch.cat(out, -1)
		out = self.dropout(out)
		outputs = self.fc(out)      
		return outputs



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
	vocab_size = len(word_to_ix)
	num_filters = 100
	filter_sizes = [3, 4, 5]
	prob = 0.1
	batch_size = 64
	lr = 0.001
	epochs = 25
	maxlen = word_sequences.shape[1]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
	model = Classifier(vocab_size, weights, maxlen, num_filters, filter_sizes, batch_size, prob, target)
	model = model.to(device)
	criterion = nn.BCEWithLogitsLoss(weight=weight[1])
	optimizer = optim.Adam(model.parameters(), lr=lr)


	# Create the training and validation data
	input_train, input_val, mask_train, mask_val, target_train, target_val, cat_train, cat_val = train_test_split(word_sequences, attention_masks, target_sequences, cats, \
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