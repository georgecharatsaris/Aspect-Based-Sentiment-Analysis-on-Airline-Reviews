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
import time



class BioTagger(nn.Module):


	def __init__(self, n_tags, prob, batch_size, device):

		super(BioTagger, self).__init__()
		self.batch_size = batch_size
		self.device = device

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(prob)
		bert_size = self.bert.config.to_dict()['hidden_size']

		self.fc = nn.Linear(bert_size, n_tags)


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

	# Import the necessary files
	texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = fileLoader('absa_dataset.txt')
	contractions = contractionsLoader('contractions.txt')
	airlines = extraLoader('airlinesNew.txt')
	aircrafts = extraLoader('aircraftsNew.txt')
	misc = extraLoader('miscNew.txt')
	airports = extraLoader('airportsNew.txt')


	# Create a dataframe
	cols = {'texts': texts, 'targets': aspects, 'categories': aspect_cats, 'attributes': aspect_attrs, 'sentiments': sentiments}
	df = pd.DataFrame(cols)
	# Create a list that contains the targets of each unique sentence
	t = df.groupby('texts', sort=False)['targets'].agg(lambda x: '  '.join(set(x)))
	# Create a list with the unique sentences	
	unique_texts = df.drop_duplicates('texts')['texts'].reset_index(drop=True)


	# Replace special tokens in the sentences
	new_texts = unique_texts.apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))


	# Replace special tokens in the targets
	new_targets = t.apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))


	# Some preprocessing of the sentences and the targets
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	x = tokenizer(new_texts.tolist(), return_token_type_ids=False, padding=True, return_tensors='pt')
	input_ids = x['input_ids']
	attention_masks = x['attention_mask']


	# After using the BERT tokenizer the word 'embeddings' splits into ['em', '##bed', '##dings'] and to keep the length of the sentence the same
	# I use the allignment trick where I consider the tag of the last sub-word ("##dings") as the tag for the original word ("embeddings") 
	y = alignment(new_texts, tokenizer)
	inputs = torch.gather(input_ids, -1, y)
	inputs = torch.where(inputs==101, torch.tensor(0), inputs)
	inputs[:, 0] = torch.tensor(101)
	

	# Detect where the target starts and ends in each sentence	
	starts, ends = targetDetection1(new_texts, new_targets)
	# Create the BIO scheme based on each target in the sentence, e.g., "Food was delicious" --> ["Food", "was", "delicious"] --> [B, O, O]
	bio_tags = bioScheme(new_texts, starts, ends, bert=True)
	# Create an index for each label (B, I, and O)
	tag_to_ix = {'<PAD>': 0, 'O': 1, 'B': 2, 'I': 3}
	tag_sequences = toSequences(bio_tags, tag_to_ix, tags=True)


	# Some important parameters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	n_tags = len(tag_to_ix)
	prob = 0.1
	batch_size = 32
	n_epochs = 4


	# Create the training and validation data
	input_train, input_val, tag_train, tag_val, mask_train, mask_val = train_test_split(inputs, tag_sequences, attention_masks, test_size=0.1, random_state=4)
	train_dataset = TensorDataset(input_train, tag_train, mask_train)
	val_dataset = TensorDataset(input_val, tag_val, mask_val)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)


	# Calculate the weights of the tags for the loss function
	weights = compute_class_weight('balanced', classes=np.unique(tag_sequences.reshape(-1).numpy()), y=tag_sequences.reshape(-1).numpy())
	weights = torch.FloatTensor(weights)


	# Set the model, the loss function, and the optimizer
	model = BioTagger(n_tags, prob, batch_size, device)
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.00005)
	criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weights)


	# Train and validate the model
	for epoch in range(n_epochs):
		start = time.time()
		loss = fit(model, train_loader, criterion, optimizer, device)
		valid_loss = evaluate(model, val_loader, criterion, device)
		end = time.time()

		elapsed_time = timer(start, end)
		print(f'Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.0f}s - loss: {loss:.4f} - val_loss: {valid_loss:.4f}')


	# Print the results of the model in the validation data
	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1], target_names=['B', 'I', 'O']))
	ax = sns.heatmap(confusion_matrix(tag_val.reshape(-1), preds.reshape(-1), labels=[2, 3, 1]), fmt='.0f', annot=True, cmap='Blues', xticklabels=['B', 'I', 'O'])
	ax.set_yticklabels(labels=['B', 'I', 'O'], rotation = 0)
	plt.show()