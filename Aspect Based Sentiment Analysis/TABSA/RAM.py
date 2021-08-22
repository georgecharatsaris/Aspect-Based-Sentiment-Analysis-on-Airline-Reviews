import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from utils import *



class RAM(nn.Module):


	def __init__(self, word_weights, gru_size, n_classes, prob, n_hops, batch_size, device):

		super(RAM, self).__init__()
		self.batch_size = batch_size
		self.device = device
		self.gru_size = gru_size
		self.n_hops = n_hops

		self.word_embeddings = nn.Embedding.from_pretrained(word_weights)
		self.word_dim = word_weights.shape[1]

		self.bigru = nn.GRU(self.word_dim, self.word_dim, bidirectional=True, batch_first=True)

		self.attention_fc = nn.Linear(2*self.word_dim + gru_size + 1, self.word_dim)
		self.gru = nn.GRU(2*self.word_dim + 1, gru_size, batch_first=True, bias=False)

		self.fc = nn.Linear(gru_size, n_classes)
		self.dropout = nn.Dropout(prob)


	def forward(self, words, positions, offsets, targets):

		batch_size = words.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.word_embeddings(words)
		target_embeddings = self.word_embeddings(targets)

		word_embeddings = self.dropout(word_embeddings)
		target_embeddings = self.dropout(target_embeddings)

		M_star, _ = self.bigru(word_embeddings)

		a = torch.einsum('bl, blk -> blk', positions, M_star)
		b = offsets.unsqueeze(2)
		M = torch.cat((a, b), dim=-1)

		e_t = torch.zeros(size=(1, self.batch_size, self.gru_size), device=self.device)

		for hop in range(self.n_hops):
			tmp = e_t
			e_t = e_t.reshape(self.batch_size, 1, self.gru_size)
			e_t = e_t.repeat(1, M.shape[1], 1)
			attention_inputs = torch.cat((M, e_t), dim=-1)

			g = self.attention_fc(attention_inputs)
			alpha = F.softmax(g, dim=1)

			i_t = torch.matmul(alpha.reshape(self.batch_size, self.word_dim, -1), M) # torch.einsum('blw, blu -> bwu', alpha, M) 
			gru_out, e_t = self.gru(i_t, tmp)

		outputs = self.fc(e_t.reshape(-1, self.gru_size))
		return outputs



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for words, positions, offsets, targets, sentiments in loader:
		words = words.to(device)
		positions = positions.to(device)
		offsets = offsets.to(device)
		targets = targets.to(device)
		sentiments = sentiments.to(device)

		model.zero_grad()
		outputs = model(words, positions, offsets, targets)
		loss = criterion(outputs, sentiments)
		loss.backward()
		optimizer.step()

		losses.append(loss)
		total += 1

	return sum(losses)/total



def evaluate(model, loader, criterion, device, predict=False):

	model.eval()

	with torch.no_grad():
		if not predict:
			losses = []
			total = 0

			for words, positions, offsets, targets, sentiments in loader:
				words = words.to(device)
				positions = positions.to(device)
				offsets = offsets.to(device)
				targets = targets.to(device)
				sentiments = sentiments.to(device)

				outputs = model(words, positions, offsets, targets)
				loss = criterion(outputs, sentiments)

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for words, positions, offsets, targets, _ in loader:
				words = words.to(device)
				positions = positions.to(device)
				offsets = offsets.to(device)
				targets = targets.to(device)

				outputs = F.softmax(model(words, positions, offsets, targets), dim=-1)
				preds += outputs.cpu().argmax(axis=-1)

		return np.array(preds)



if __name__ == "__main__":

	# Import the necessary files
	texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = fileLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/absa_dataset.txt')
	contractions = contractionsLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/contractions.txt')
	airlines = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/airlinesNew.txt')
	aircrafts = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/aircraftsNew.txt')
	misc = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/miscNew.txt')
	airports = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/airportsNew.txt')


	# Create the dataframe
	cols = {'texts': texts, 'targets': targets, 'aspects': aspects, 'categories': aspect_cats, 'attributes': aspect_attrs, 'sentiments': sentiments}
	df = pd.DataFrame(cols)


	# Replace the special tokens in the unique sentences
	new_texts = df['texts'].apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))


	# Replace the special tokens in the targets
	new_targets = df['targets'].apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))


	# Some preprocessing of the unique texts and targets
	new_texts = new_texts.apply(textPreprocessing, args=(contractions, ))
	new_targets = new_targets.apply(textPreprocessing, args=(contractions, ))


	# Create the vocabulary
	word_to_ix = prepareVocab(new_texts)
	# Detect the aspect in each sentence
	starts, ends = targetDetection2(new_texts, new_targets)
	# Create the word and target sequences as well as the position and offsets embeddings
	word_sequences = toSequences(new_texts, word_to_ix)
	target_sequences = toSequences(new_targets, word_to_ix)
	positions, offsets = poSequences(new_texts, starts, ends, new_targets)

	# Set the paremeters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vocab_size = len(word_to_ix) + 1
	prob = 0.1
	gru_size = 300
	batch_size = 64
	lr = 0.001
	n_epochs = 25
	n_classes = 3
	n_hops = 11


	# Load the domain-specific word embeddings
	w2v = KeyedVectors.load('C:/Users/gxara/Documents/Master Thesis/Datasets/word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	weights = embedding_weight.to(device)


	# Create the training and validation data
	sentiments = torch.LongTensor(sentiments)
	train_sequence, val_sequence, train_pos, val_pos, train_offset, val_offset, \
	train_target, val_target, train_sentiment, val_sentiment = train_test_split(word_sequences, positions, offsets, target_sequences, sentiments, test_size=0.1, \
																				random_state=4)
	train_dataset = TensorDataset(train_sequence, train_pos, train_offset, train_target, train_sentiment)
	val_dataset = TensorDataset(val_sequence, val_pos, val_offset, val_target, val_sentiment)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Compute the weights for the three sentiment to be used in the loss function
	weight = compute_class_weight('balanced', classes=np.unique(train_sentiment), y=train_sentiment.numpy())
	weight = torch.FloatTensor(weight)
	weight = weight.to(device)


	# Set the model, the optimizer, and the loss function
	model = RAM(weights, gru_size, n_classes, prob, n_hops, batch_size, device)
	model = model.to(device)
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
	criterion = nn.CrossEntropyLoss(weight=weight)


	# Train the model
	for epoch in range(n_epochs):
	    start = time.time()
	    loss = fit(model, train_loader, criterion, optimizer, device)
	    valid_loss = evaluate(model, val_loader, criterion, device)
	    end = time.time()

	    elapsed_time = timer(start, end)
	    print(f'Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.0f}s - loss: {loss:.4f} - val_loss: {valid_loss:.4f}')


	# Print the results of the model in the validation data
	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(val_sentiment.numpy(), preds.reshape(-1)))
	sns.heatmap(confusion_matrix(val_sentiment.numpy(), preds.reshape(-1)), cmap='Blues', annot=True, fmt='.0f')
	plt.show()