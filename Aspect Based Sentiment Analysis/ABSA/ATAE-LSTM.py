import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim



class ATAE(nn.Module):


	def __init__(self, word_weights, aspect_size, lstm_size, n_classes, batch_size, prob, device):

		super(ATAE, self).__init__()
		self.batch_size = batch_size
		self.lstm_size = lstm_size
		self.device = device

		self.U_w = nn.Embedding.from_pretrained(word_weights)
		word_dim = word_weights.shape[1]
		self.U_a = nn.Embedding(aspect_size, word_dim)
		nn.init.uniform_(self.U_a.weight, a=-0.1, b=0.1)

		self.word_gru = nn.GRU(2*word_dim, lstm_size, batch_first=True)

		self.W_h = nn.Linear(lstm_size, lstm_size, bias=False)
		self.W_u = nn.Linear(word_dim, lstm_size, bias=False)
		self.w = nn.Linear(2*lstm_size, 1, bias=False)

		self.W_p = nn.Linear(lstm_size, lstm_size, bias=False)
		self.W_x = nn.Linear(lstm_size, lstm_size, bias=False)

		self.W_s = nn.Linear(lstm_size, n_classes)
		self.dropout = nn.Dropout(prob)


	def forward(self, words, aspects):

		batch_size = words.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.U_w(words))
		aspect_embeddings = self.U_a(aspects)

		wmaxlen = word_embeddings.shape[1]
		amaxlen = aspect_embeddings.shape[1]

		if wmaxlen%amaxlen != 0:
			word_embeddings = word_embeddings.repeat(1, amaxlen, 1)
			aspect_embeddings = aspect_embeddings.repeat(1, wmaxlen, 1)
		else:
			scale = wmaxlen//amaxlen
			aspect_embeddings = aspect_embeddings.repeat(1, scale, 1)

		gru_in = torch.cat((word_embeddings, aspect_embeddings), dim=-1)
		gru_out, h_N = self.word_gru(gru_in)

		M = torch.cat((self.W_h(gru_out), self.W_u(aspect_embeddings)), dim=-1)
		a = F.softmax(self.w(M), dim=1)
		r = torch.matmul(gru_out.reshape(self.batch_size, self.lstm_size, -1), a)

		h_star = torch.tanh(self.W_p(r.reshape(self.batch_size, 1, -1)) + self.W_x(h_N.reshape(self.batch_size, 1, self.lstm_size)))

		out = self.W_s(h_star.reshape(-1, self.lstm_size))
		return out



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs, aspects, sentiments in loader:
		inputs = inputs.to(device)
		aspects = aspects.to(device)
		sentiments = sentiments.to(device)

		model.zero_grad()
		outputs = model(inputs, aspects)
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

			for inputs, aspects, sentiments in loader:
				inputs = inputs.to(device)
				aspects = aspects.to(device)
				sentiments = sentiments.to(device)

				outputs = model(inputs, aspects)
				loss = criterion(outputs, sentiments)

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for inputs, aspects, _ in loader:
				inputs = inputs.to(device)
				aspects = aspects.to(device)

				outputs = F.softmax(model(inputs, aspects), dim=-1)
				preds += outputs.cpu().argmax(axis=-1)

			return np.array(preds)



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


	# Replace the special tokens and process the texts
	new_texts = df['texts'].apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))
	new_texts = new_texts.apply(textPreprocessing, args=(contractions, ))


	# Prepare the vocabulary of the words and the aspects
	word_to_ix = prepareVocab(new_texts)
	aspect_to_ix = prepareVocab(aspects)


	# Prepare the inputs to the model
	word_sequences = toSequences(new_texts, word_to_ix)
	aspect_sequences = toSequences(aspects, aspect_to_ix)


	# Set the parameters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vocab_size = len(word_to_ix) + 1
	aspect_size = len(aspect_to_ix) + 1
	prob = 0.1
	lstm_size = 50
	batch_size = 64
	lr = 0.001
	n_epochs = 25
	n_classes = 3


	# Create the training and validation data
	sentiments = torch.LongTensor(sentiments)
	train_sequence, val_sequence, train_aspect, val_aspect, train_sentiment, val_sentiment = train_test_split(word_sequences, aspect_sequences, sentiments, test_size=0.1, \
																											  random_state=4)
	train_dataset = TensorDataset(train_sequence, train_aspect, train_sentiment)
	val_dataset = TensorDataset(val_sequence, val_aspect, val_sentiment)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Calculate the weights of each sentiment in the training data
	weights = compute_class_weight('balanced', classes=np.unique(train_sentiment), y=train_sentiment.numpy())
	weights = torch.FloatTensor(weights)
	weight = weights.to(device)


	# Load the domain-specific word embeddings
	w2v = KeyedVectors.load('word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	word_weights = embedding_weight.to(device)


	# Set the model, the optimizer, and the loss function
	model = ATAE(word_weights, aspect_size, lstm_size, n_classes, batch_size, prob, device)
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


	# Print the results on the validation data
	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(val_sentiment.numpy(), preds.reshape(-1)))
	sns.heatmap(confusion_matrix(val_sentiment.numpy(), preds.reshape(-1)), cmap='Blues', annot=True, fmt='.0f')
	plt.show()		