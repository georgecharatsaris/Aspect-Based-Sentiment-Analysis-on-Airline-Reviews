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

Xuezhe Ma and Eduard Hovy

### Abstract

State-of-the-art sequence labeling systems
traditionally require large amounts of taskspecific
knowledge in the form of handcrafted
features and data pre-processing.
In this paper, we introduce a novel neutral
network architecture that benefits from
both word- and character-level representations
automatically, by using combination
of bidirectional LSTM, CNN and CRF.
Our system is truly end-to-end, requiring
no feature engineering or data preprocessing,
thus making it applicable to
a wide range of sequence labeling tasks.
We evaluate our system on two data sets
for two sequence labeling tasks — Penn
Treebank WSJ corpus for part-of-speech
(POS) tagging and CoNLL 2003 corpus
for named entity recognition (NER).
We obtain state-of-the-art performance on
both datasets—97.55% accuracy for POS
tagging and 91.21% F1 for NER.

[[Paper]](https://arxiv.org/abs/1603.01354) [[Code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Target%20Detection/CNN-BiLSTM-CRF.py)

### Run Example

```
cd Target Detection
python CNN-BiLSTM-CRF.py
```

## BERT:

### Authors

Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova

### Abstract

We introduce a new language representation
model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language representation
models (Peters et al., 2018a; Radford
et al., 2018), BERT is designed to pretrain
deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a result,
the pre-trained BERT model can be finetuned
with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substantial taskspecific
architecture modifications.
BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results
on eleven natural language processing
tasks, including pushing the GLUE score to
80.5% (7.7% point absolute improvement),
MultiNLI accuracy to 86.7% (4.6% absolute
improvement), SQuAD v1.1 question answering
Test F1 to 93.2 (1.5 point absolute improvement)
and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).

[[Paper]](https://arxiv.org/abs/1810.04805) [[Code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Target%20Detection/BERT.py)

### Run Example

```
cd Target Detection
python BERT.py
```

# Category Detection

## Support Vector Machine (SVM):

### Authors

Zhifei Zhang and Jian-Yun NieYey, Hongling Wang

### Abstract

This paper describes the system we submitted
to In-domain ABSA subtask of SemEval 2015
shared task on aspect-based sentiment analysis that includes aspect category detection and
sentiment polarity classification. For the aspect category detection, we combined an SVM
classifier with implicit aspect indicators. For
the sentiment polarity classification, we combined an SVM classifier with a lexicon-based
polarity classifier. Our system outperforms the
baselines on both the laptop and restaurant domains and ranks above average on the laptop
domain.

[[Paper]](https://aclanthology.org/S15-2131.pdf) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Category%20Detection/SVM.py)

### Run Example

```
cd Category Detection
python SVM.py
```

## Convolutional Neural Network (CNN):

### Authors

Yoon Kim

### Abstract

We report on a series of experiments with
convolutional neural networks (CNN)
trained on top of pre-trained word vectors
for sentence-level classification tasks.
We show that a simple CNN with little
hyperparameter tuning and static vectors
achieves excellent results on multiple
benchmarks. Learning task-specific
vectors through fine-tuning offers further
gains in performance. We additionally
propose a simple modification to the architecture
to allow for the use of both
task-specific and static vectors. The CNN
models discussed herein improve upon the
state of the art on 4 out of 7 tasks, which
include sentiment analysis and question
classification.

[[Paper]](https://arxiv.org/abs/1408.5882) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Category%20Detection/CNN.py)

### Run Example

```
cd Category Detection
python CNN.py
```

## Bidirection Gated Recurrent Unit (BiGRU):

[[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Category%20Detection/BiGRU.py)

### Run Example

```
cd Category Detection
python BiGRU.py
```

## (HAHNN):

### Authors

Jader Abreu, Luis Fred, David Macedo, and Cleber Zanchettin

### Abstract

Document classification is a challenging task with important
applications. The deep learning approaches to the problem have gained
much attention recently. Despite the progress, the proposed models do
not incorporate the knowledge of the document structure in the architec-
ture efficiently and not take into account the contexting importance of
words and sentences. In this paper, we propose a new approach based on
a combination of convolutional neural networks, gated recurrent units,
and attention mechanisms for document classification tasks. The main
contribution of this work is the use of convolution layers to extract more
meaningful, generalizable and abstract features by the hierarchical rep-
resentation. The proposed method in this

[[Paper]](https://arxiv.org/abs/1901.06610) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Category%20Detection/HAHNN.py)

### Run Example

```
cd Category Detection
python HAHNN.py
```

## (BERT):

[[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Category%20Detection/BERT.py)

### Run Example

```
cd Category Detection
python BERT.py
```

# Aspect Based Sentiment Analysis (ABSA)

## SVM:

[[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/ABSA/SVM.py)

```
cd Aspect Based Sentiment Analysis/ABSA
python SVM.py
```

## (ATAE-LSTM):

### Authors

Yequan Wang and Minlie Huang and Li Zhao and Xiaoyan Zhu

### Abstract

Aspect-level sentiment classification is a finegrained
task in sentiment analysis. Since it
provides more complete and in-depth results,
aspect-level sentiment analysis has received
much attention these years. In this paper, we
reveal that the sentiment polarity of a sentence
is not only determined by the content but is
also highly related to the concerned aspect.
For instance, “The appetizers are ok, but the
service is slow.”, for aspect taste, the polarity
is positive while for service, the polarity
is negative. Therefore, it is worthwhile to explore
the connection between an aspect and
the content of a sentence. To this end, we
propose an Attention-based Long Short-Term
Memory Network for aspect-level sentiment
classification. The attention mechanism can
concentrate on different parts of a sentence
when different aspects are taken as input. We
experiment on the SemEval 2014 dataset and
results show that our model achieves state-ofthe-
art performance on aspect-level sentiment
classification.

[[Paper]](https://aclanthology.org/D16-1058/) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/ABSA/ATAE-LSTM.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/ABSA
python ATAE-LSTM.py
```

## (CrossNet):

### Authors

Chang Xu, C´ecile Paris, Surya Nepal, and Ross Sparks

### Abstract

In stance classification, the target on which
the stance is made defines the boundary of
the task, and a classifier is usually trained
for prediction on the same target. In this
work, we explore the potential for generalizing
classifiers between different targets,
and propose a neural model that can apply
what has been learned from a source target
to a destination target. We show that our
model can find useful information shared
between relevant targets which improves
generalization in certain scenarios.

[[Paper]](https://aclanthology.org/P18-2123/) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/ABSA/CrossNet.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/ABSA
python CrossNet.py
```

## (HEAT):

### Authors
Jiajun Cheng, Shenglin Zhao, Jiani Zhang, Irwin King, Xin Zhang, Hui Wang

### Abstract

Aspect-level sentiment classification is a fine-grained sentiment
analysis task, which aims to predict the sentiment of a text in different
aspects. One key point of this task is to allocate the appropriate
sentiment words for the given aspect. Recent work exploits attention
neural networks to allocate sentiment words and achieves
the state-of-the-art performance. However, the prior work only
attends to the sentiment information and ignores the aspect-related
information in the text, which may cause mismatching between
the sentiment words and the aspects when an unrelated sentiment
word is semantically meaningful for the given aspect. To solve this
problem, we propose a HiErarchical ATtention (HEAT) network for
aspect-level sentiment classification. The HEAT network contains
a hierarchical attention module, consisting of aspect attention and
sentiment attention. The aspect attention extracts the aspect-related
information to guide the sentiment attention to better allocate
aspect-specific sentiment words of the text. Moreover, the HEAT
network supports to extract the aspect terms together with aspectlevel
sentiment classification by introducing the Bernoulli attention
mechanism. To verify the proposed method, we conduct experiments
on restaurant and laptop review data sets from SemEval
at both the sentence level and the review level. The experimental
results show that our model better allocates appropriate sentiment
expressions for a given aspect benefiting from the guidance of aspect
terms. Moreover, our method achieves better performance on
aspect-level sentiment classification than state-of-the-art models.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/3132847.3133037) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/ABSA/HEAT.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/ABSA
python HEAT.py
```

# Targeted Aspect Based Sentiment Analysis (TABSA)

## (SVM):

[[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/TABSA/SVM.py)

```
cd Aspect Based Sentiment Analysis/TABSA
python SVM.py
```

## (TC-LSTM):

### Authors

Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu

### Abstract

Target-dependent sentiment classification remains a challenge: modeling the semantic relatedness
of a target with its context words in a sentence. Different context words have different
influences on determining the sentiment polarity of a sentence towards the target. Therefore, it
is desirable to integrate the connections between target word and context words when building a
learning system. In this paper, we develop two target dependent long short-term memory (LSTM)
models, where target information is automatically taken into account. We evaluate our methods
on a benchmark dataset from Twitter. Empirical results show that modeling sentence representation
with standard LSTM does not perform well. Incorporating target information into LSTM
can significantly boost the classification accuracy. The target-dependent LSTM models achieve
state-of-the-art performances without using syntactic parser or external sentiment lexicons.

[[Paper]](https://aclanthology.org/C16-1311/) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/TABSA/TC-LSTM.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/TABSA
python TC-LSTM.py
```

## (IAN):

### Authors

Dehong Ma, Sujian Li1, Xiaodong Zhang, Houfeng Wang

### Abstract

Aspect-level sentiment classification aims at identifying
the sentiment polarity of specific target in
its context. Previous approaches have realized
the importance of targets in sentiment classification
and developed various methods with the goal
of precisely modeling their contexts via generating
target-specific representations. However, these
studies always ignore the separate modeling of targets.
In this paper, we argue that both targets and
contexts deserve special treatment and need to be
learned their own representations via interactive
learning. Then, we propose the interactive attention
networks (IAN) to interactively learn attentions in
the contexts and targets, and generate the representations
for targets and contexts separately. With this
design, the IAN model can well represent a target
and its collocative context, which is helpful to sentiment
classification. Experimental results on SemEval
2014 Datasets demonstrate the effectiveness
of our model.

[[Paper]](https://arxiv.org/abs/1709.00893) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/TABSA/IAN.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/TABSA
python IAN.py
```

## (MemNet):

### Authors

Duyu Tang, Bing Qin, Ting Liu

### Abstract

We introduce a deep memory network for
aspect level sentiment classification. Unlike
feature-based SVM and sequential neural
models such as LSTM, this approach explicitly
captures the importance of each context
word when inferring the sentiment polarity
of an aspect. Such importance degree and
text representation are calculated with multiple
computational layers, each of which is a
neural attention model over an external memory.
Experiments on laptop and restaurant
datasets demonstrate that our approach performs
comparable to state-of-art feature based
SVM system, and substantially better than
LSTM and attention-based LSTM architectures.
On both datasets we show that multiple
computational layers could improve the
performance. Moreover, our approach is also
fast. The deep memory network with 9 layers
is 15 times faster than LSTM with a CPU
implementation.

[[Paper]](https://arxiv.org/abs/1605.08900) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/TABSA/MemNet.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/TABSA
python MemNet.py
```

## (RAM):

### Authors

Peng Chen, Zhongqian Sun, Lidong Bing, Wei Yang

### Abstract

We propose a novel framework based on
neural networks to identify the sentiment
of opinion targets in a comment/review.
Our framework adopts multiple-attention
mechanism to capture sentiment features
separated by a long distance, so that it
is more robust against irrelevant information.
The results of multiple attentions
are non-linearly combined with a recurrent
neural network, which strengthens the
expressive power of our model for handling
more complications. The weightedmemory
mechanism not only helps us
avoid the labor-intensive feature engineering
work, but also provides a tailor-made
memory for different opinion targets of a
sentence. We examine the merit of our
model on four datasets: two are from SemEval2014,
i.e. reviews of restaurants and
laptops; a twitter dataset, for testing its
performance on social media data; and a
Chinese news comment dataset, for testing
its language sensitivity. The experimental
results show that our model consistently
outperforms the state-of-the-art methods
on different types of data.

[[Paper]](https://aclanthology.org/D17-1047/) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/TABSA/RAM.py)

### Run Example

```
cd Aspect Based Sentiment Analysis/TABSA
python RAM.py
```

# Targeted Aspect Based Sentiment Analysis with Aspect Category

## (ABSA-BERT):

### Authors

Chi Sun, Luyao Huang, Xipeng Qiu

### Abstract

Aspect-based sentiment analysis (ABSA),
which aims to identify fine-grained opinion
polarity towards a specific aspect, is a challenging
subtask of sentiment analysis (SA).
In this paper, we construct an auxiliary sentence
from the aspect and convert ABSA to a
sentence-pair classification task, such as question
answering (QA) and natural language inference
(NLI). We fine-tune the pre-trained
model from BERT and achieve new state-ofthe-
art results on SentiHood and SemEval-
2014 Task 4 datasets.

[[Paper]](https://arxiv.org/abs/1903.09588) [[code]](https://github.com/georgecharatsaris/Aspect-Based-Sentiment-Analysis-on-Airline-Reviews/blob/main/Aspect%20Based%20Sentiment%20Analysis/ABSA-BERT.py)

### Run Example

```
cd Aspect Based Sentiment Analysis
python ABSA-BERT.py
```