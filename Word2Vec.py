from Models.utils import embeddingMatrix
from gensim.models import Word2Vec
import pandas as pd
from utils import *
import re



def trainVectors(texts, min_df, size, sg):

	"""Returns the bag of words, the dictionary of the corpus, and the w2v vectors of the words in the dictionary.
		Arguments:
			texts: A list of lists of processed documents (at least without punctuation).
			min_df : The minimum number of documents that contain a word.
			size: The size of w2v vectors.
			sg: Training algorithm: 1 for skip-gram, 0 for CBOW. 
		Returns:
			w2v: The w2v model built on the corpus.
	"""

	bow = [[token for token in text.split()] for text in texts] 
	w2v = Word2Vec(bow, vector_size=size, min_count=min_df, sg=sg)
	return w2v



def noNumbers(text):

	for word in text.split():
		if re.search(r'\d+', word):
			text = re.sub(word, 'digit', text)

	return text



if __name__ == "__main__":

	df = pd.read_excel('reviews airline.xlsx')
	texts = df['review_text'].astype('str')


	contractions = contractionsLoader('contractions.txt')
	new_texts = texts.apply(textPreprocessing)
	new_texts = new_texts.apply(preprocessing, args=(contractions, ))
	new_texts = new_texts.apply(noNumbers)


	w2v = trainVectors(new_texts, min_df=1, size=300, sg=0)
	word_vectors = w2v.wv
	word_vectors.save('word2vec.kv')