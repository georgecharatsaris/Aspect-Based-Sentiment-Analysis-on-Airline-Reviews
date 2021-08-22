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



class HEAT(nn.Module):


	def __init__(self, word_weights, aspect_size, aspect_dim, n_classes, prob, batch_size, device):

		super(HEAT, self).__init__()
		self.batch_size = batch_size
		self.device = device

		self.word_embeddings = nn.Embedding.from_pretrained(word_weights)
		self.aspect_embeddings = nn.Embedding(aspect_size, aspect_dim)
		nn.init.uniform_(self.aspect_embeddings.weight, a=-0.1, b=0.1)
		self.dropout = nn.Dropout(prob)

		word_dim = word_weights.shape[1]
		self.bilstm_a = nn.LSTM(word_dim, aspect_dim, bidirectional=True, batch_first=True)
		self.bilstm_s = nn.LSTM(word_dim, aspect_dim, bidirectional=True, batch_first=True)

		self.fc_a = nn.Linear(3*aspect_dim, aspect_dim)
		self.fc_s = nn.Linear(5*aspect_dim, aspect_dim)

		self.u_a = nn.Linear(aspect_dim, 1, bias=False)
		nn.init.xavier_uniform_(self.u_a.weight)
		self.u_s = nn.Linear(aspect_dim, 1, bias=False)
		nn.init.xavier_uniform_(self.u_s.weight)

		self.fc = nn.Linear(3*aspect_dim, n_classes)


	def forward(self, inputs, aspects, M):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.word_embeddings(inputs))
		aspect_embeddings = self.dropout(self.aspect_embeddings(aspects))

		H_a, _ = self.bilstm_a(word_embeddings)
		H_s, _ = self.bilstm_s(word_embeddings)

		# Aspect Attention
		input_a = torch.cat((H_a, aspect_embeddings.repeat(1, H_a.shape[1], 1)), dim=-1)
		g_a = self.u_a(self.fc_a(input_a))
		a_a = F.softmax(g_a, dim=1)
		v_a = torch.einsum('blo, bld -> bod', a_a, H_a)

		# Sentiment Attention
		input_b = torch.cat((H_a, v_a.repeat(1, H_a.shape[1], 1), aspect_embeddings.repeat(1, H_a.shape[1], 1)), dim=-1)
		g_s = self.u_s(self.fc_s(input_b))
		m_l = torch.einsum('bll, blo -> blo', M, a_a)
		a_s = F.softmax(torch.einsum('blo, blo -> blo', m_l, g_s), dim=-1)
		v_s = torch.einsum('blo, bld -> bod', a_s, H_s)

		y = self.fc(torch.cat((v_s.squeeze(), aspect_embeddings.squeeze()), dim=-1))
		return y



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs, aspects, M, sentiments in loader:
		inputs = inputs.to(device)
		aspects = aspects.to(device)
		M = M.to(device)
		sentiments = sentiments.to(device)

		model.zero_grad()
		outputs = model(inputs, aspects, M)
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

			for inputs, aspects, M, sentiments in loader:
				inputs = inputs.to(device)
				aspects = aspects.to(device)
				M = M.to(device)
				sentiments = sentiments.to(device)

				outputs = model(inputs, aspects, M)
				loss = criterion(outputs, sentiments)

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for inputs, aspects, M, _ in loader:
				inputs = inputs.to(device)
				aspects = aspects.to(device)
				M = M.to(device)

				outputs = F.softmax(model(inputs, aspects, M), dim=-1)
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
	maxlen = max([len(text.split()) for text in new_texts])
	vocab_size = len(word_to_ix) + 1
	aspect_size = len(aspect_to_ix) + 1
	aspect_dim = 100
	n_classes = 3 
	prob = 0.1
	batch_size = 64
	lr = 0.01
	size = len(df)
	n_epochs = 25


	# Create the training and validation data
	M = posMatrix(size, maxlen, new_texts)
	sentiments = torch.LongTensor(sentiments)
	train_sequence, val_sequence, train_aspect, val_aspect, train_M, test_M, train_sentiment, val_sentiment = train_test_split(word_sequences, aspect_sequences, M, \
																															   sentiments, test_size=0.1, random_state=4)
	train_dataset = TensorDataset(train_sequence, train_aspect, train_M, train_sentiment)
	val_dataset = TensorDataset(val_sequence, val_aspect, test_M, val_sentiment)
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
	model = HEAT(word_weights, aspect_size, aspect_dim, n_classes, prob, batch_size, device)
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