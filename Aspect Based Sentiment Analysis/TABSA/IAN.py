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



class IANet(nn.Module):


	def __init__(self, word_weights, maxlen, lstm_size, n_classes, batch_size, prob, device):

		super(IANet, self).__init__()
		self.word_dim = word_weights.shape[1]
		self.lstm_size = lstm_size
		self.batch_size = batch_size
		self.maxlen = maxlen
		self.device = device

		self.word_embeddings = nn.Embedding.from_pretrained(word_weights)

		self.word_gru = nn.GRU(self.word_dim, lstm_size, batch_first=True, bidirectional=True)
		self.target_gru = nn.GRU(self.word_dim, lstm_size, batch_first=True, bidirectional=True)

		self.fc1 = nn.Linear(2*lstm_size, lstm_size)
		self.fc2 = nn.Linear(lstm_size, self.maxlen)

		self.fc = nn.Linear(4*lstm_size, n_classes)
		self.dropout = nn.Dropout(prob)


	def forward(self, words, targets):

		batch_size = words.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.word_embeddings(words))
		target_embeddings = self.dropout(self.word_embeddings(targets))

		words_out, _ = self.word_gru(word_embeddings)
		targets_out, _ = self.target_gru(target_embeddings)

		c_avg = torch.mean(words_out, dim=1, keepdim=True)
		t_avg = torch.mean(targets_out, dim=1, keepdim=True)

		W_a = torch.Tensor(size=(self.batch_size, 2*self.lstm_size, 2*self.lstm_size)).to(device)
		nn.init.xavier_uniform_(W_a)
		b_a = torch.zeros(size=(self.batch_size, 1), device=self.device)
		W_a.requires_grad, b_a.requires_grad = True, True

		w = torch.matmul(words_out, W_a)
		w = torch.matmul(w, t_avg.reshape(self.batch_size, -1, 1))
		wgamma = torch.tanh(w + b_a.unsqueeze(1).expand(self.batch_size, words_out.shape[1], 1))

		alpha = F.softmax(wgamma, dim=1)

		t = torch.matmul(targets_out, W_a)
		t = torch.matmul(t, c_avg.reshape(self.batch_size, -1, 1))
		tgamma = torch.tanh(t + b_a.unsqueeze(1).expand(self.batch_size, targets_out.shape[1], 1))

		beta = F.softmax(tgamma, dim=1)

		c_r = torch.sum(torch.matmul(alpha.reshape(self.batch_size, -1, words_out.shape[1]), words_out), dim=1, keepdim=True)
		t_r =torch.sum(torch.matmul(beta.reshape(self.batch_size, -1, targets_out.shape[1]), targets_out), dim=1, keepdim=True)

		d = torch.cat((c_r, t_r), dim=-1)
		out = self.fc(d.squeeze())
		return out



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for words, targets, sentiments in loader:
		words = words.to(device)
		targets = targets.to(device)
		sentiments = sentiments.to(device)

		model.zero_grad()
		outputs = model(words, targets)
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

			for words, targets, sentiments in loader:
				words = words.to(device)
				targets = targets.to(device)
				sentiments = sentiments.to(device)

				outputs = model(words, targets)
				loss = criterion(outputs, sentiments)

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for words, targets, _ in loader:
				words = words.to(device)
				targets = targets.to(device)

				outputs = F.softmax(model(words, targets), dim=-1)
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
	# Create the word and target sequences
	word_sequences = toSequences(new_texts, word_to_ix)
	target_sequences = toSequences(new_targets, word_to_ix)


	# Set the paremeters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vocab_size = len(word_to_ix) + 1
	lstm_size = 100
	n_classes = 3
	batch_size = 64
	prob = 0.1
	lr = 0.001
	n_epochs = 25
	maxlen = word_sequences.shape[1]


	# Load the domain-specific word embeddings
	w2v = KeyedVectors.load('C:/Users/gxara/Documents/Master Thesis/Datasets/word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	weights = embedding_weight.to(device)


	# Create the training and validation data
	sentiments = torch.LongTensor(sentiments)
	train_sequence, val_sequence, train_target, val_target, train_sentiment, val_sentiment = train_test_split(word_sequences, target_sequences, sentiments, test_size=0.1, \
																											  random_state=4)
	train_dataset = TensorDataset(train_sequence, train_target, train_sentiment)
	val_dataset = TensorDataset(val_sequence, val_target, val_sentiment)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Compute the weights for the three sentiment to be used in the loss function
	weight = compute_class_weight('balanced', classes=np.unique(train_sentiment), y=train_sentiment.numpy())
	weight = torch.FloatTensor(weight)
	weight = weight.to(device)


	# Set the model, the optimizer, and the loss function
	model = IANet(weights, maxlen, lstm_size, n_classes, batch_size, prob, device)
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