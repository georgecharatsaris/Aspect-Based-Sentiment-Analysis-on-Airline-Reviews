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



class TC_LSTM(nn.Module):


	def __init__(self, weights, vocab_size, n_classes, batch_size, prob):
		super(TC_LSTM, self).__init__()
		self.batch_size = batch_size

		self.word_embeddings = nn.Embedding.from_pretrained(weights)
		word_dim = weights.shape[1]
		self.word_dim = word_dim

		self.lstmL = nn.LSTM(2*word_dim, word_dim, batch_first=True)
		self.lstmR = nn.LSTM(2*word_dim, word_dim, batch_first=True)

		self.fc = nn.Linear(2*word_dim, n_classes)
		self.dropout = nn.Dropout(prob)


	def forward(self, left_inputs, right_inputs, targets):

		batch_size = targets.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_vectors_left = self.dropout(self.word_embeddings(left_inputs))
		word_vectors_right = self.dropout(self.word_embeddings(right_inputs))
		aspect_vectors = self.dropout(self.word_embeddings(targets))

		u_target = torch.mean(aspect_vectors, dim=1, keepdim=True)

		inputs_left = torch.cat((word_vectors_left, u_target.repeat(1, word_vectors_left.shape[1], 1)), dim=2)
		inputs_right = torch.cat((word_vectors_right, u_target.repeat(1, word_vectors_right.shape[1], 1)), dim=2)

		output_left, (hidden_left, _) = self.lstmL(inputs_left)
		output_right, (hidden_right, _) = self.lstmR(inputs_right)
		lstm_output = torch.cat((hidden_left.reshape(-1, 1, self.word_dim), hidden_right.reshape(-1, 1, self.word_dim)), dim=-1)

		y = self.fc(lstm_output.squeeze(1))
		return y



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs_left, inputs_right, targets, sentiments in loader:
		inputs_left = inputs_left.to(device)
		inputs_right = inputs_right.to(device)
		targets = targets.to(device)
		sentiments = sentiments.to(device)

		model.zero_grad()
		outputs = model(inputs_left, inputs_right, targets)
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

			for inputs_left, inputs_right, targets, sentiments in loader:
				inputs_left = inputs_left.to(device)
				inputs_right = inputs_right.to(device)
				targets = targets.to(device)				
				sentiments = sentiments.to(device)

				outputs = model(inputs_left, inputs_right, targets)
				loss = criterion(outputs, sentiments)
				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for inputs_left, inputs_right, targets, _ in loader:
				inputs_left = inputs_left.to(device)
				inputs_right = inputs_right.to(device)
				targets = targets.to(device)

				outputs = F.softmax(model(inputs_left, inputs_right, targets), dim=-1)
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
	# Split the sentence into the left and the right part
	texts_left, texts_right = leftToRight(new_texts, starts, ends)
	# Create the left and right word embeddings as well as the target sequences
	left_sequences = toSequences(texts_left, word_to_ix)
	right_sequences = toSequences(texts_right, word_to_ix)
	target_sequences = toSequences(new_targets, word_to_ix)


	# Set the paremeters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vocab_size = len(word_to_ix) + 1
	batch_size = 64
	aspect_dim = 100
	n_classes = 3
	lr = 0.001
	n_epochs = 25
	prob = 0.1


	# Load the domain-specific word embeddings
	w2v = KeyedVectors.load('C:/Users/gxara/Documents/Master Thesis/Datasets/word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	weights = embedding_weight.to(device)


	# Create the training and validation data
	sentiments = torch.LongTensor(sentiments)
	train_left, val_left, train_right, val_right, train_target, val_target, train_sentiment, val_sentiment = train_test_split(left_sequences, right_sequences, \
																															  target_sequences, sentiments, test_size=0.1, \
																															  random_state=4)
	train_dataset = TensorDataset(train_left, train_right, train_target, train_sentiment)
	val_dataset = TensorDataset(val_left, val_right, val_target, val_sentiment)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Compute the weights for the three sentiment to be used in the loss function
	weight = compute_class_weight('balanced', classes=np.unique(train_sentiment), y=train_sentiment.numpy())
	weight = torch.FloatTensor(weight)
	weight = weight.to(device)


	# Set the model, the optimizer, and the loss function
	model = TC_LSTM(weights, vocab_size, n_classes, batch_size, prob)
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