import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from utils import *



class Classifier(nn.Module):


	def __init__(self, prob, batch_size):

		super(Classifier, self).__init__()
		self.batch_size = batch_size

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(prob)
		bert_size = self.bert.config.to_dict()['hidden_size']

		self.fc = nn.Linear(bert_size, 1)


	def forward(self, inputs, masks):

		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size

		x = self.bert(inputs, masks)
		x = self.dropout((x['pooler_output']))
		x = self.fc(x)
		return x



def fit(model, loader, criterion, optimizer, device):

	model.train()
	losses = []
	total = 0

	for inputs, masks, targets in loader:
		inputs = inputs.to(device)
		masks = masks.to(device)
		targets = targets.to(device)

		model.zero_grad()
		outputs = model(inputs, masks)
		loss = criterion(outputs, targets.reshape(-1, 1))
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
			for inputs, masks, targets in loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				targets = targets.to(device)

				outputs = model(inputs, masks)
				loss = criterion(outputs, targets.reshape(-1, 1))

				losses.append(loss)
				total += 1

			return sum(losses)/total
		else:
			preds = []

			for inputs, masks, targets in loader:
				inputs = inputs.to(device)
				masks = masks.to(device)
				targets = targets.to(device)

				outputs = model(inputs, masks)
				outputs = torch.sigmoid(outputs)

				preds += torch.round(outputs.cpu())

			return pad_sequence(preds, batch_first=True)



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


	# Replace the special tokens in the targets
	new_targets = df['targets'].apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))

	# Split the data into training and test data
	txt_train, txt_test, trg_train, trg_test, asp_train, asp_test, sent_train, sent_test = train_test_split(new_texts.to_list(), new_targets.to_list(), aspects, sentiments, \
																											test_size=0.1, random_state=4)
	# Create sub-texts from the original texts for the binary classification using the text, the target, the aspect, and the sentiment
	# For example, "The food was delicious" yields:
	# 1) ['The food was delicious', 'The polarity of the "FOOD#QUALITY" aspect of food is positive.', 1]
	# 2) ['The food was delicious', 'The polarity of the "FOOD#QUALITY" aspect of food is negative.', 0]
	# 3) ['The food was delicious', 'The polarity of the "FOOD#QUALITY" aspect of food is neutral.', 0]
	# Tthe instance that holds gets the "1" value (YES) while the others get the "0" value ("NO")
	pos_train, neg_train, neu_train = polarityQA(txt_train, trg_train, asp_train, sent_train)
	pos_test, neg_test, neu_test = polarityQA(txt_test, trg_test, asp_test, sent_test)


	# Create the training dataframe
	pos_train_df = pd.DataFrame(pos_train, columns=['Reviews', 'Polarity', 'Yes/No'])
	neg_train_df = pd.DataFrame(neg_train, columns=['Reviews', 'Polarity', 'Yes/No'])
	neu_train_df = pd.DataFrame(neu_train, columns=['Reviews', 'Polarity', 'Yes/No'])
	train_df = pd.concat([neg_train_df, neu_train_df, pos_train_df], axis=0).sample(frac = 1).reset_index(drop=True)


	# Create the test dataframes
	pos_test_df = pd.DataFrame(pos_test, columns=['Reviews', 'Polarity', 'Yes/No'])
	neg_test_df = pd.DataFrame(neg_test, columns=['Reviews', 'Polarity', 'Yes/No'])
	neu_test_df = pd.DataFrame(neu_test, columns=['Reviews', 'Polarity', 'Yes/No'])
	test_df = pd.concat([neg_test_df, neu_test_df, pos_test_df], axis=0).sort_index().reset_index(drop=True)


	# Initialize BERT tokenizer and use it for each of the two dataframes created above
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	x = tokenizer(train_df['Reviews'].tolist(), train_df['Polarity'].tolist(), return_tensors='pt', padding=True, return_token_type_ids=False)
	y = tokenizer(test_df['Reviews'].tolist(), test_df['Polarity'].tolist(), return_tensors='pt', padding=True, return_token_type_ids=False)
	inputs_train = x['input_ids']
	masks_train = x['attention_mask']
	inputs_test = y['input_ids']
	masks_test = y['attention_mask']


	# Set the parameters
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	prob = 0.25
	batch_size = 32
	lr = 0.00005
	n_epochs = 4


	# Compute the weights of the sentiments to be used in the loss function
	weight = compute_class_weight('balanced', classes=np.unique(train_df['Yes/No']), y=train_df['Yes/No'])
	weight = torch.FloatTensor(weight)
	weight = weight.to(device)	


	# Set the model, the optimizer, and the loss function
	model = Classifier(prob, batch_size)
	model = model.to(device)
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
	criterion = nn.BCEWithLogitsLoss(weight=weight[1])


	# Create the training and validation dataset
	bin_train = torch.FloatTensor(train_df['Yes/No'])
	bin_test = torch.FloatTensor(test_df['Yes/No'])
	train_dataset = TensorDataset(inputs_train, masks_train, bin_train)
	test_dataset = TensorDataset(inputs_test, masks_test, bin_test)
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(test_dataset, batch_size=batch_size)


	# Train the model for the binary classification
	for epoch in range(n_epochs):
	    start = time.time()
	    loss = fit(model, train_loader, criterion, optimizer, device)
	    val_loss = evaluate(model, val_loader, criterion, device)
	    end = time.time()

	    elapsed_time = timer(start, end)
	    print(f'Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.0f}s - loss: {loss:.4f} - val_loss: {val_loss:.4f}')


	# Print the results of the binary classification in the test data
	preds = evaluate(model, val_loader, criterion, device, predict=True)
	print(classification_report(bin_test.numpy(), preds.reshape(-1)))
	sns.heatmap(confusion_matrix(bin_test.numpy(), preds.reshape(-1)), cmap='Blues', annot=True, cbar=False, fmt='.0f')


	# Find all the true positive test examples for each of the three sentiments
	a = test_df[['Polarity', 'Yes/No']]
	rpos = a[a['Polarity'].str.contains('positive')]['Yes/No']
	rneu = a[a['Polarity'].str.contains('negative')]['Yes/No']
	rneg = a[a['Polarity'].str.contains('neutral')]['Yes/No']


	# Calculate the correctly predicted sentiments and print their number
	pospred, negpred, neupred = 0, 0, 0

	for (i, j), k in zip(a.iterrows(), preds):
	    if j['Polarity'].find('positive') != -1 and j['Yes/No'] == 1 and k == 1:
	        pospred += 1
	    elif j['Polarity'].find('negative') != -1 and j['Yes/No'] == 1 and k == 1:
	        negpred += 1
	    elif j['Polarity'].find('neutral') != -1 and j['Yes/No'] == 1 and k == 1:
	        neupred += 1

	print(pospred, negpred, neupred)	