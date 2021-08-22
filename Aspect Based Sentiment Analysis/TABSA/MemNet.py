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



class MemNet(nn.Module):
    
	def __init__(self, word_weights, n_classes, n_hops, prob, batch_size):

		super(MemNet, self).__init__()
		self.batch_size = batch_size
		self.n_hops = n_hops

		self.word_embeddings = nn.Embedding.from_pretrained(word_weights)
		self.dropout = nn.Dropout(prob)

		self.word_dim = word_weights.shape[1]       
		self.attention_fc = nn.Linear(2*self.word_dim, 1)

		self.transformation_fc = nn.Linear(self.word_dim, self.word_dim)

		self.output_fc = nn.Linear(self.word_dim, n_classes)

	def forward(self, inputs, targets):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.dropout(self.word_embeddings(inputs))
		target_embeddings = self.word_embeddings(targets)
		u_aspect = torch.mean(target_embeddings, dim=1, keepdim=True)

		for i in range(self.n_hops):           
			attention_input = torch.cat((word_embeddings, u_aspect.repeat(1, word_embeddings.shape[1], 1)), dim=-1)
			g = torch.tanh(self.attention_fc(attention_input))
			a = F.softmax(g, dim=1)
			vec = torch.sum(torch.matmul(a.view(a.shape[0], a.shape[2], a.shape[1]), word_embeddings), dim=1, keepdim=True)

			transformed_aspect = self.transformation_fc(target_embeddings)

			target_embeddings = torch.add(vec, transformed_aspect)
			u_aspect = torch.mean(target_embeddings, dim=1, keepdim=True)

		outputs = self.output_fc(u_aspect.view((-1, self.word_dim)))
		return outputs



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
	prob = 0.1
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
	model = MemNet(weights, n_classes, n_hops, prob, batch_size)
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