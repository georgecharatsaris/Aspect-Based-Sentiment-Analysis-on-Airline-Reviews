import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
from nltk.corpus import stopwords
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk



# ntlk.download('stopwords')



def grid_search(svm, d, X_train, y_train, score='f1_macro'):

	grid = GridSearchCV(estimator=svm, param_grid=d, scoring=score)
	grid.fit(X_train, y_train)

	return grid.best_params_



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


	# Create an onehot encoding for each aspect category
	onehot = df['aspects'].str.get_dummies(sep=' ').values


	# Process the stop words and initialize the tfidf vectorizer
	stop_words = stopwords.words('english') 
	stop_words = stopWords(stop_words, contractions)
	tfidf = TfidfVectorizer(stop_words=stop_words)


	# Create the sentence representations and concatenate each of them with the respective aspect onehot encoding 
	dtm = tfidf.fit_transform(new_texts)
	inputs = np.concatenate((dtm.toarray(), onehot), axis=-1)


	# Set the parameters for the grid search and create the training and test data
	d = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1]}
	X_train, X_test, y_train, y_test = train_test_split(inputs, sentiments, test_size=0.1, random_state=4)


	# Compute the weights of the three sentiments in the training data
	weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
	weights_dict = {0: weights[0], 1: weights[1], 2: weights[2]}


	# Run grid search and train the model using the best parameters
	params = grid_search(SVC(class_weight=weights_dict), d, X_train, y_train)
	svm = SVC(C=params['C'], gamma=params['gamma'], class_weight=weights_dict)
	svm.fit(X_train, y_train)


	# Print the results of the model in the test data
	preds = svm.predict(X_test)
	print(classification_report(y_test, preds))
	sns.heatmap(confusion_matrix(y_test, preds), cmap='Blues', annot=True, fmt='.0f', cbar=False)