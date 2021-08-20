import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from Models.utils import *
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence



class BioTagger(nn.Module):


	def __init__(self, weights, lstm_size, n_tags, batch_size, prob):

		super(BioTagger, self).__init__()
		self.batch_size = batch_size

		self.embeddings = nn.Embedding.from_pretrained(weights)
		self.dropout = nn.Dropout(prob)
		emb_dim = weights.shape[1]

		self.bi_lstm = nn.LSTM(emb_dim, lstm_size, bidirectional=True, batch_first=True)

		self.fc = nn.Linear(2*lstm_size, n_tags, bias=False)
		self.crf = CRF(n_tags, batch_first=True)


	def forward(self, inputs, tags, masks, train=True):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		word_embeddings = self.embeddings(inputs)
		word_embeddings = self.dropout(word_embeddings)

		bi_lstm_out, _ = self.bi_lstm(word_embeddings)

		emissions = self.fc(bi_lstm_out)

		if train:
			loss = self.crf(emissions, tags, masks)
			return loss
		else:
			out = self.crf.decode(emissions, masks)
			max_cols = max([len(batch) for batch in tags])
			out = torch.FloatTensor([batch + [0] * (max_cols - len(batch)) for batch in out])        
			return out



def fit(model, train_loader, optimizer):

	model.train()
	losses = []
	total = 0

	for inputs, targets, masks in train_loader:
		inputs = inputs.to(device)
		masks = masks.to(device)
		targets = targets.to(device)

		model.zero_grad()
		loss = -1 * model(inputs, targets, masks)
		loss.backward()
		optimizer.step()

		losses.append(loss)
		total += 1

	return sum(losses)/total



def evaluate(model, val_loader):

	model.eval()
	losses = []
	total = 0

	with torch.no_grad():
		for inputs, targets, masks in val_loader:
			inputs = inputs.to(device)
			masks = masks.to(device)
			targets = targets.to(device)

			loss = -1 * model(inputs, targets, masks)

			losses.append(loss)
			total += 1

	return sum(losses)/total



def predict(model, loader):

	model.eval()

	with torch.no_grad():
		tmp = []

		for inputs, tags, masks in loader:
			inputs = inputs.to(device)
			masks = masks.to(device)
			tags = tags.to(device)

			preds = model(inputs, tags, masks, train=False)
			tmp += preds

	return pad_sequence(tmp, batch_first=True)



if __name__ == '__main__':

	texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = fileLoader('absa_dataset.txt')
	contractions = contractionsLoader('contractions.txt')
	airlines = extraLoader('airlines.txt')
	aircrafts = extraLoader('aircrafts.txt')
	misc = extraLoader('misc.txt')
	airports = extraLoader('airports.txt')


	cols = {'texts': texts, 'targets': aspects, 'categories': aspect_cats, 'attributes': aspect_attrs, 'sentiments': sentiments}
	df = pd.DataFrame(cols)
	t = df.groupby('texts', sort=False)['Targets'].agg(lambda x: '  '.join(set(x)))
	unique_texts = df.drop_duplicates('texts')['texts'].reset_index(drop=True)


	new_texts = unique_texts.apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))


	new_targets = t.apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))


	new_texts = new_texts.apply(textPreprocessing, args=(contractions, ))
	new_targets = new_targets.apply(textPreprocessing, args=(contractions, ))


	word_to_ix = prepareVocab(new_texts)
	starts, ends = targetDetection(new_texts, new_targets)
	bio_tags = bioScheme(new_texts, starts, ends)
	tag_to_ix = prepareVocab(bio_tags, tags=True)


	word_sequences = toSequences(new_texts, word_to_ix)
	tag_sequences = toSequences(bio_tags, tag_to_ix, tags=True)
	attention_masks = attentionMasks(new_texts)


	vocab_size = len(word_to_ix)
	lstm_size = 100
	n_tags = len(tag_to_ix)
	batch_size = 64
	prob = 0.1
	n_epochs = 100


	w2v = KeyedVectors.load('word2vec.kv')
	embedding_matrix = w2vMatrix(vocab_size, w2v, word_to_ix)
	embedding_weight = torch.FloatTensor(embedding_matrix)
	weights = embedding_weight.to(device)


	model = BioTagger(weights, lstm_size, n_tags, batch_size, prob)
	optimizer = optim.SGD([p for p in model.parameters()], lr=0.00005)


	char_train, char_val, word_train, word_val, tag_train, tag_val, mask_train, mask_val = train_test_split(char_sequences, word_sequences, tag_sequences, attention_masks, \
																											test_size=0.1, random_state=4)
	train_dataset = TensorDataset(char_train, word_train, tag_train, mask_train)
	val_dataset = TensorDataset(char_val, word_val, tag_val, mask_val)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	for epoch in range(n_epochs):
		start = time.time()
		loss = fit(model, train_loader, optimizer)
		valid_loss = evaluate(model, val_loader)
		end = time.time()

		elapsed_time = timer(start, end)
		print(f'Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.0f}s - loss: {loss:.4f} - val_loss: {valid_loss:.4f}')


	preds = predict(model, val_loader)
	print(classification_report(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1], target_names=['B', 'I', 'O']))
	ax = sns.heatmap(confusion_matrix(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1]), cmap='Blues', annot=True, fmt='.0f', xticklabels=['B', 'I', 'O'])
	ax.set_yticklabels(labels=['B', 'I', 'O'], rotation = 0)
	plt.show()