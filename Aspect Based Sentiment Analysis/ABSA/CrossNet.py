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



class CrossNet(nn.Module):


	def __init__(self, word_weights, aspect_size, lstm_size, n_classes, batch_size, prob):

		super(CrossNet, self).__init__()
		self.batch_size = batch_size

		self.embedding = nn.Embedding.from_pretrained(word_weights)
		emb_dim = word_weights.shape[1]
		self.aspect_embedding = nn.Embedding(aspect_size, emb_dim)
		nn.init.uniform_(self.aspect_embedding.weight, a=-0.1, b=0.1)

		self.bi_lstm1 = nn.LSTM(emb_dim, lstm_size, bidirectional=True, batch_first=True)
		self.bi_lstm2 = nn.LSTM(emb_dim, lstm_size, bidirectional=True, batch_first=True)

		self.fc1 = nn.Linear(2*lstm_size, emb_dim)
		self.fc2 = nn.Linear(emb_dim, 1)

		self.fc3 = nn.Linear(2*lstm_size, n_classes)
		self.dropout = nn.Dropout(prob)


	def forward(self, inputs, aspects):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.embedding(inputs)
		aspect_embeddings = self.aspect_embedding(aspects)
		word_embeddings = self.dropout(word_embeddings)

		out1, (hidden1, cell1) = self.bi_lstm1(aspect_embeddings)
		out2, _ = self.bi_lstm2(word_embeddings, (hidden1, cell1))

		c = self.fc2(torch.sigmoid(self.fc1(out2)))	 
		a = F.softmax(c, dim=1)
		A = torch.matmul(a.reshape(self.batch_size, 1, -1), out2).squeeze(dim=1)

		out = self.fc3(A)
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
	texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = fileLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/absa_dataset.txt')
	contractions = contractionsLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/contractions.txt')
	airlines = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/airlinesNew.txt')
	aircrafts = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/aircraftsNew.txt')
	misc = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/miscNew.txt')
	airports = extraLoader('C:/Users/gxara/Documents/Master Thesis/Datasets/airportsNew.txt')


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
	lstm_size = 50
	n_classes = 3
	batch_size = 64
	n_epochs = 25
	lr = 0.001
	prob = 0.1


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
	w2v = KeyedVectors.load('C:/Users/gxara/Documents/Master Thesis/Datasets/word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	word_weights = embedding_weight.to(device)


	# Set the model, the optimizer, and the loss function
	model = CrossNet(word_weights, aspect_size, lstm_size, n_classes, batch_size, prob)
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