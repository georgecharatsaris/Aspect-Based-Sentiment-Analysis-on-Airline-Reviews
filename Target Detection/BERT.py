import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import train_test_split



class BioTagger(nn.Module):


	def __init__(self, prob, batch_size, device):

		super(BertClassifier, self).__init__()
		self.batch_size = batch_size
		self.device = device

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(prob)
		bert_size = self.bert.config.to_dict()['hidden_size']

		self.fc = nn.Linear(bert_size, 4)


	def forward(self, inputs, masks):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		x = self.bert(inputs, masks)
		x = self.dropout(x['last_hidden_state'])

		out = self.fc(x)
		return out



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs, tags, masks in loader:
		inputs = inputs.to(device)
		masks = masks.to(device)
		tags = tags.to(device)

		model.zero_grad()
		outputs = model(inputs, masks)
		loss = criterion(outputs.reshape(-1, 4), tags.reshape(-1))
		loss.backward()
		optimizer.step()

		losses.append(loss)
		total += 1

	return sum(losses)/total



def evaluate(model, loader, criterion, device, predict=False):

	model.eval()
	losses = []
	total = 0

	with torch.no_grad():
		if not predict:
			for inputs, tags, masks in loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				tags = tags.to(device)

				outputs = model(inputs, masks)
				loss = criterion(outputs.reshape(-1, 4), tags.reshape(-1))

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for inputs, tags, masks in loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				tags = tags.to(device)

				outputs = model(inputs, masks)
				outputs = F.softmax(outputs, dim=-1)

				preds += outputs.cpu().argmax(-1)

			return pad_sequence(preds, batch_first=True)



if __name__ == "__main__":

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


	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	x = tokenizer(new_texts.tolist(), return_token_type_ids=False, padding=True, return_tensors='pt')
	input_ids = x['input_ids']
	attention_masks = x['attention_mask']


	y = alignment(new_texts, tokenizer)
	inputs = torch.gather(input_ids, -1, y)
	inputs = torch.where(inputs==101, torch.tensor(0), inputs)
	inputs[:, 0] = torch.tensor(101)
	

	orig_to_token = alignment(new_texts, tokenizer)
	starts, ends = targetDetection(new_texts, new_targets)
	bio_tags = bioScheme(new_texts, starts, ends, bert=True)
	tag_to_ix = {'<PAD>': 0, 'O': 1, 'B': 2, 'I': 3}
	tag_sequences = toSequences(bio_tags, tag_to_ix, tags=True)


	n_tags = len(tag_to_ix)
	prob = 0.1
	batch_size = 32
	n_epochs = 4


	input_train, input_val, tag_train, tag_val, mask_train, mask_val = train_test_split(inputs, tag_sequences, attention_masks, test_size=0.1, random_state=4)
	train_dataset = TensorDataset(input_train, tag_train, mask_train)
	val_dataset = TensorDataset(input_val, tag_val, mask_val)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)
	weights = compute_class_weight('balanced', np.unique(tag_sequences.reshape(-1).numpy()), tag_sequences.reshape(-1).numpy())
	weights = torch.FloatTensor(weights)


	model = BioTagger(bert, n_tags, prob, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=0.00005)
	criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weights)


	for epoch in range(n_epochs):
		start = time.time()
		loss = fit(model, train_loader, criterion, optimizer, device)
		valid_loss = evaluate(model, val_loader, criterion, device)
		end = time.time()

		elapsed_time = timer(start, end)
		print(f'Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.0f}s - loss: {loss:.4f} - val_loss: {valid_loss:.4f}')


	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1], target_names=['B', 'I', 'O']))
	ax = sns.heatmap(confusion_matrix(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1]), fmt='.0f', annot=True, cmap='Blues', xticklabels=['B', 'I', 'O'])
	ax.set_yticklabels(labels=['B', 'I', 'O'], rotation = 0)
	plt.show()