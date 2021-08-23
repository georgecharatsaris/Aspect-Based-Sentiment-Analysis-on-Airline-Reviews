import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from gensim.models import KeyedVectors
import pickle



nltk.download('stopwords')



def grid_search(svm, d, X_train, y_train, score='f1'):

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
	# Create a list of the entities in each unique sentence
	c = df.groupby('texts', sort=False)['categories'].agg(lambda x: '  '.join(set(x)))
	# Create a list of the attributes in each unique sentence
	a = df.groupby('texts', sort=False)['attributes'].agg(lambda x: '  '.join(set(x)))
	# Create a list of the targets in each unique sentence
	t = df.groupby('texts', sort=False)['targets'].agg(lambda x: '  '.join(set(x)))
	# Create a list of unique sentences
	unique_texts = df.drop_duplicates('texts')['texts'].reset_index(drop=True)


	# Replace the special tokens in the unique sentences
	new_texts = unique_texts.apply(replaceToken, args=(airlines, 'airline'))
	new_texts = new_texts.apply(replaceToken, args=(airports, 'airport'))
	new_texts = new_texts.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_texts = new_texts.apply(replaceToken, args=(misc, 'misc'))


	# Replace the special tokens in the targets
	new_targets = t.apply(replaceToken, args=(airlines, 'airline'))
	new_targets = new_targets.apply(replaceToken, args=(airports, 'airport'))
	new_targets = new_targets.apply(replaceToken, args=(aircrafts, 'aircraft'))
	new_targets = new_targets.apply(replaceToken, args=(misc, 'misc'))


	# Some preprocessing of the unique texts and targets
	new_texts = new_texts.apply(textPreprocessing, args=(contractions, ))
	new_targets = new_targets.apply(textPreprocessing, args=(contractions, ))


	# Load and process the stop words for the tfidf vectorizer
	stop_words = stopwords.words('english') 
	stop_words = stopWords(stop_words, contractions)


	# Set the tfidf vectorizer and the parameters for the grid search
	tfidf = TfidfVectorizer(stop_words=stop_words)
	d = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1]}
	target = True


	# Create the document term matrix
	dtm = tfidf.fit_transform(new_texts)
	# Define the vocabulary size
	vocab_size = dtm.shape[1]
	# Load the word embeddings
	word_vectors = KeyedVectors.load('word2vec.kv')


	# Create the input containing the sentence representation plus the target information in each row (when target == True) or use only the sentence
	# representation taken by the tfidf vectorize (when target == False)
	if target:
		embeddings = []

		for targets in new_targets:
		    tmp = []
		    
		    if len(targets.split()) > 1:
		        for target in targets.split():
		            if target.lower() in word_vectors.key_to_index.keys() and target.lower() != 'null': 
		                tmp.append(word_vectors[target.lower()])
		        
		        embeddings.append(np.mean(tmp, 0))
		    else:
		        for target in targets.split():
		            if target.lower() in word_vectors.key_to_index.keys() and target.lower() != 'null': 
		                embeddings.append(word_vectors[target.lower()])
		            else:
		                embeddings.append(np.zeros(shape=(300, )))

		X = np.append(dtm.toarray(), embeddings, axis=-1)
	else:
		X = dtm.toarray()


	# Define onehot representations of the entities
	entities = c.str.get_dummies(sep=' ').values
	airline = entities[:, 0]
	ambience = entities[:, 1]
	connectivity = entities[:, 2]
	entertainment = entities[:, 3]
	foodANDdrinks = entities[:, 4]
	seat = entities[:, 5]
	service = entities[:, 6]


	# Define onehot representations of the attributes
	attributes = a.str.get_dummies(sep=' ').values
	cabin = attributes[:, 0]
	comfort = attributes[:, 1]
	general = attributes[:, 2]
	ground = attributes[:, 3]
	options = attributes[:, 4]
	prices = attributes[:, 5]
	quality = attributes[:, 6]
	schedule = attributes[:, 7]
	wifi = attributes[:, 8]


	# Train for the "AIRLINE" entity
	y = torch.FloatTensor(airline)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)
	weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
	weights_dict = {0: weights[0], 1: weights[1]}


	# Run the grid search and train the model using the best parameters
	params = grid_search(SVC(class_weight=weights_dict), d, X_train, y_train)
	svm = SVC(C=params['C'], gamma=params['gamma'], class_weight=weights_dict)
	svm.fit(X_train, y_train)


	# Print the results of the model in the validation data
	preds = svm.predict(X_test)
	print(classification_report(y_test, preds))
	sns.heatmap(confusion_matrix(y_test, preds), cmap='Blues', annot=True, fmt='.0f', cbar=False)
	plt.show()