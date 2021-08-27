# Aspect-Based-Sentiment-Analysis-on-Airline-Reviews

This repository was created for my master thesis project. I split the Aspect Based Sentiment Analysis task into three sub-tasks, Target
Detection, Category Detection, and Aspect Based Sentiment Analysis. The first task aims at identifying the target words in the sentence, the
second at categorizing the sentence as an aspect category from a predefined set of entities/attributes, and the last at finding the sentiment
of the sentence towards the target (TABSA), the aspect category (ABSA), or both of them (ABSA-BERT). 

# Dataset

The dataset was created by me. First, I scraped airline reviews from Skytrax website [[Link]](https://skytraxratings.com/) and then I randomly
chose some of them, split them into sentences. Then, I annotated the subjective sentences based on my perspective. An example of the annotation
is as follows:

The $t$ was delicious.  
food  
FOOD#QUALITY  
1

The first line is the sentence with the target word been replaced by the symbol $t$, the second one is the target word, the third is the aspect
category of the target word and the last is the sentiment of the sentence towards the target word. If a sentence contains an implicit target, I
use the word "NULL" as the target word.

# Useful Files

- airlines:names of the airlines included in the dataset
- airports:names of the airports included in the dataset
- aircrafts:names of the aircrafts included in the dataset
- misc:names of the products, companies, foods, etc., included in the dataset
- Word2Vec embeddings:domain-specific word embeddings trained on the dataset
- contractions: contractions dictionary 

# Installation

```
git clone https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews.git
cd Aspect-Based-Sentiment-Analysis-on-Airline-Reviews
pip install -r requirements.txt
```

# Target Detection

## BiLSTM+CRF:

### Authors

Athanasios Giannakopoulos, Claudiu Musat, Andreea Hossmann

### Abstract

Aspect Term Extraction (ATE) identifies
opinionated aspect terms in texts and is
one of the tasks in the SemEval Aspect
Based Sentiment Analysis (ABSA)
contest. The small amount of available
datasets for supervised ATE and the
costly human annotation for aspect term
labelling give rise to the need for unsupervised
ATE. In this paper, we introduce
an architecture that achieves top-ranking
performance for supervised ATE. Moreover,
it can be used efficiently as feature
extractor and classifier for unsupervised
ATE. Our second contribution is a
method to automatically construct datasets
for ATE. We train a classifier on our automatically
labelled datasets and evaluate it
on the human annotated SemEval ABSA
test sets. Compared to a strong rule-based
baseline, we obtain a dramatically higher
F-score and attain precision values above
80%. Our unsupervised method beats the
supervised ABSA baseline from SemEval,
while preserving high precision scores.

[[Paper]](https://aclanthology.org/W17-5224/) [[Code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Target%20Detection/BiLSTM-CRF.py)

### Run Example

```
cd Target Detection
python BiLSTM-CRF.py
```

## CNN+BiLSTM+CRF:

### Authors

D. Kuang, J. Choo, H. Park

### Abstract

Nonnegative matrix factorization (NMF) approximates a nonnegative matrix by the product of 
two low-rank nonnegative matrices. Since it gives semantically meaningful result that is 
easily interpretable in clustering applications, NMF has been widely used as a clustering 
method especially for document data, and as a topic modeling method.We describe several 
fundamental facts of NMF and introduce its optimization framework called block coordinate 
descent. In the context of clustering, our framework provides a flexible way to extend NMF 
such as the sparse NMF and the weakly-supervised NMF. The former provides succinct 
representations for better interpretations while the latter flexibly incorporate extra 
information and user feedback in NMF, which effectively works as the basis for the visual 
analytic topic modeling system that we present.Using real-world text data sets, we present 
quantitative experimental results showing the superiority of our framework from the 
following aspects: fast convergence, high clustering accuracy, sparse representation, 
consistent output, and user interactivity. In addition, we present a visual analytic system 
called UTOPIAN (User-driven Topic modeling based on Interactive NMF) and show several usage 
scenarios.Overall, our book chapter cover the broad spectrum of NMF in the context of 
clustering and topic modeling, from fundamental algorithmic behaviors to practical visual 
analytics systems.

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-09259-1_7) [[Code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Target%20Detection/CNN-BiLSTM-CRF.py)

### Run Example

```
cd Target Detection
python CNN-BiLSTM-CRF.py
```

## BERT:

### Authors

Akash Srivastava, Charles Sutton

### Abstract

Topic models are one of the most popular methods for learning representations of text, but a 
major challenge is that any change to the topic model requires mathematically deriving a new 
inference algorithm. A promising approach to address this problem is autoencoding variational 
Bayes (AEVB), but it has proven difficult to apply to topic models in practice. We present 
what is to our knowledge the first effective AEVB based inference method for latent Dirichlet 
allocation (LDA), which we call Autoencoded Variational Inference For Topic Model (AVITM). 
This model tackles the problems caused for AEVB by the Dirichlet prior and by component 
collapsing. We find that AVITM matches traditional methods in accuracy with much better 
inference time. Indeed, because of the inference network, we find that it is unnecessary to 
pay the computational cost of running variational optimization on test data. Because AVITM 
is black box, it is readily applied to new topic models. As a dramatic illustration of this, 
we present a new topic model called ProdLDA, that replaces the mixture model in LDA with a 
product of experts. By changing only one line of code from LDA, we find that ProdLDA yields 
much more interpretable topics, even if LDA is trained via collapsed Gibbs sampling.

[[Paper]](https://arxiv.org/abs/1703.01488) [[Code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Target%20Detection/BERT.py)

### Run Example

```
cd Target Detection
python BERT.py
```

# Category Detection

## Support Vector Machine (SVM):

### Authors

Rui Wangy, Xuemeng Huy, Deyu Zhouy, Yulan Hex, Yuxuan Xiongy, Chenchen Yey, Haiyang Xuz

### Abstract

Recent years have witnessed a surge of interests of using neural topic models for automatic
topic extraction from text, since they avoid the complicated mathematical derivations for
model inference as in traditional topic models such as Latent Dirichlet Allocation (LDA).
However, these models either typically assume improper prior (e.g. Gaussian or Logistic 
Normal) over latent topic space or could not infer topic distribution for a given document. 
To address these limitations, we propose a neural topic modeling approach, called 
Bidirectional Adversarial Topic (BAT) model, which represents the first attempt of applying 
bidirectional adversarial training for neural topic modeling. The proposed BAT builds a 
twoway projection between the document-topic distribution and the document-word distribution.
It uses a generator to capture the semantic patterns from texts and an encoder for topic 
inference. Furthermore, to incorporate word relatedness information, the Bidirectional 
Adversarial Topic model with Gaussian (Gaussian-BAT) is extended from BAT. To verify the 
effectiveness of BAT and Gaussian- BAT, three benchmark corpora are used in our experiments. 
The experimental results show that BAT and Gaussian-BAT obtain more coherent topics, 
outperforming several competitive baselines. Moreover, when performing text clustering based 
on the extracted topics, our models outperform all the baselines, with more significant 
improvements achieved by Gaussian-BAT where an increase of near 6% is observed in accuracy.

[[Paper]](https://arxiv.org/abs/2004.12331) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/bat.py)

### Run Example

```
cd models
python bat.py
```

## Convolutional Neural Network (CNN):

### Authors

Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei

### Abstract

Topic modeling analyzes documents to learn meaningful patterns of words. However, existing 
topic models fail to learn interpretable topics when working with large and heavy-tailed 
vocabularies. To this end, we develop the embedded topic model (ETM), a generative model of 
documents that marries traditional topic models with word embeddings. In particular, it 
models each word with a categorical distribution whose natural parameter is the inner 
product between a word embedding and an embedding of its assigned topic. To fit the ETM, we 
develop an efficient amortized variational inference algorithm. The ETM discovers 
interpretable topics even with large vocabularies that include rare words and stop words. 
It outperforms existing document models, such as latent Dirichlet allocation, in terms of 
both topic quality and predictive performance.

[[Paper]](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00325) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/etm.py)

### Run Example

```
cd models
python etm.py
```

## Bidirection Gated Recurrent Unit (BiGRU):

### Authors

Federico Bianchi, Silvia Terragni, Dirk Hovy

### Abstract

Topic models extract meaningful groups of words from documents, allowing for a better 
understanding of data. However, the solutions are often not coherent enough, and thus harder 
to interpret. Coherence can be improved by adding more contextual knowledge to the model. 
Recently, neural topic models have become available, while BERT-based representations have 
further pushed the state of the art of neural models in general. We combine pre-trained 
representations and neural topic models. Pre-trained BERT sentence embeddings indeed support 
the generation of more meaningful and coherent topics than either standard LDA or existing 
neural topic models. Results on four datasets show that our approach effectively increases 
topic coherence.

[[Paper]](https://arxiv.org/abs/2004.03974) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/ctm.py) [[GitHub]](https://github.com/MilaNLProc/contextualized-topic-models)

### Run Example

```
cd models
python ctm.py
```

## (HAHNN):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (BERT):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

# Aspect Based Sentiment Analysis (ABSA)

## SVM:

```
cd models
python dtm.py
```

## (ATAE-LSTM):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (CrossNet):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (HEAT):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

# Targeted Aspect Based Sentiment Analysis (TABSA)

## (SVM):

```
cd models
python dtm.py
```

## (TC-LSTM):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (IAN):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (MemNet):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

## (RAM):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```

# Targeted Aspect Based Sentiment Analysis with Aspect Category

## (ABSA-BERT):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```