import re
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np



def fileLoader(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        texts, targets, aspects, aspect_cats, aspect_attrs, sentiments = [], [], [], [], [], []
        polarity = {'-1': 0, '0': 1, '1': 2}

        for i in range(0, len(lines), 4):
            text = lines[i].strip()
            target = lines[i + 1].strip()
            aspect = lines[i + 2].strip()
            aspect_cat, aspect_attr = lines[i + 2].strip().split('#')
            sentiment = lines[i + 3].strip()

            new_text = text.replace('$t$', target)

            texts.append(new_text)
            targets.append(target)
            aspects.append(aspect)
            aspect_cats.append(aspect_cat)
            aspect_attrs.append(aspect_attr)
            sentiments.append(polarity[sentiment])

        
    return texts, targets, aspects, aspect_cats, aspect_attrs, sentiments



def contractionsLoader(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        l = f.readlines()
        contractions = {}

        for i in range(len(l)):
            key, value = l[i].split(':')
            contractions[key.strip()] = value.strip()

    return contractions



def extraLoader(filepath):

    texts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        file = f.readlines()

        for i in range(len(file)):
            texts.append(file[i].strip())

    return texts



def replaceToken(text, tokens, word):
    
    for token in tokens:
        if re.search(token, text):
            return re.sub(token, word, text)

    return text



def textPreprocessing(text, contractions):
    
    for word in text.split():
        if word.lower() in contractions.keys():
            text = re.sub(word, contractions[word.lower()], text)

    text = re.sub(r'[^\w\s]', '', text)

    for word in text.split():
        if re.search(r'\d+', word) != None:
            text = re.sub(word, 'DIGIT', text)

    return text.lower()



def prepareVocab(inputs, tags=False):

    to_ix = {'<pad>': 0}

    if not tags:    
        for text in inputs:
            for word in text.lower().split():
                if word not in to_ix:
                    to_ix[word] = len(to_ix)
    else:
        for tags in inputs:
            for tag in tags:
                if tag not in to_ix:
                    to_ix[tag] = len(to_ix)

    return to_ix



def targetDetection(unique_texts, targets):

    starts, ends = [], []

    for text, target in zip(unique_texts, targets):
        if re.search('  ', target.lower()):
            k = target.split('  ')
            tmp1, tmp2 = [], []

            for i in k:
                if i != 'null':
                    start = text.lower().find(i)
                    end = start + len(i)
                    tmp1.append(start)
                    tmp2.append(end)
                else:
                    tmp1.append(0)
                    tmp2.append(0)

            starts.append(tmp1)
            ends.append(tmp2)
        else:
            if target != 'null':
                start = text.lower().find(target.lower())
                end = start + len(target.lower())
                starts.append([start])
                ends.append([end])
            else:
                starts.append([0])
                ends.append([0])

    return starts, ends



def prepareVocab(inputs, tags=False):

    to_ix = {'<pad>': 0}
    
    if not tags:    
        for text in inputs:
            for word in text.lower().split():
                if word not in to_ix:
                    to_ix[word] = len(to_ix)
    else:
        for tags in inputs:
            for tag in tags:
                if tag not in to_ix:
                    to_ix[tag] = len(to_ix)
                
    return to_ix



def toSequences(inputs, to_ix, tags=False):

    idxs = []

    if not tags:
        for i in inputs:
            idxs.append(torch.LongTensor([to_ix[j] for j in i.lower().split()]))
    else:
        for i in inputs:
            idxs.append(torch.LongTensor([to_ix[j] for j in i]))

    return pad_sequence(idxs, batch_first=True)



def bioScheme(texts, starts, ends, bert=False):

    bio_tags = []

    for text, start, end in zip(texts, starts, ends):
        index = 0

        if not bert:
            sent_tags = []
        else:
            sent_tags = ['<PAD>']

        previous_flag = 'O'

        min_start, max_start = min(start), max(start)
        min_end, max_end = min(end), max(end)

        for word in text.split():
            position = index + len(word)

            if position < max_start:
                if position < min_start or position > min_end:
                    sent_tags.append('O')
                    previous_flag = 'O'
                    index += len(word) + 1 
                else:
                    if previous_flag == 'O':
                        sent_tags.append('B')
                        previous_flag = 'B'
                        index += len(word) + 1 
                    else:
                        sent_tags.append('I')
                        index += len(word) + 1
            else:
                if position > max_end:
                    sent_tags.append('O')
                    previous_flag = 'O'
                    index += len(word) + 1 
                else:
                    if previous_flag == 'O':
                        sent_tags.append('B')
                        previous_flag = 'B'
                        index += len(word) + 1 
                    else:
                        sent_tags.append('I')
                        index += len(word) + 1

        if bert:
            sent_tags.append('<PAD>') 

        bio_tags.append(sent_tags)

    return bio_tags



def attentionMasks(inputs, tokenizer=None, bert=False):

    idxs = []
    
    if not bert:
        for i in inputs:
            idxs.append(torch.tensor([1]*len(i.split()), dtype=torch.uint8))
    else:
        for i in inputs:
            idxs.append(torch.tensor([1] + [1]*len(tokenizer.tokenize(i) + [1]), dtype=torch.uint8))
        
    return pad_sequence(idxs, batch_first=True)



def charSequences(texts, to_ix):

    idxs = []

    for i, text in enumerate(texts):
        idxs.append(torch.LongTensor([to_ix[char] for char in text.lower()]))

    return pad_sequence(idxs, batch_first=True)



def timer(start, end):
    return end - start



def alignment(texts, tokenizer):

    orig_to_token = []

    for text in texts:
        tmp = []
        tmp.append(0)

        for word in text.split():
            k = len(tokenizer.wordpiece_tokenizer.tokenize(word))
            tmp.append(tmp[-1] + k)

        tmp.append(tmp[-1] + 1)

        orig_to_token.append(torch.LongTensor(tmp))

    return pad_sequence(orig_to_token, batch_first=True)



def stopWords(stop_words, contractions):

    for i in range(len(stop_words)):
        if stop_words[i] in contractions:
            stop_words[i] = contractions[stop_words[i]]

    stop_words = stop_words + (['could', 'might', 'must', 'need', 'shall', 'would'])

    return stop_words



def embeddingsLoader(filepath):

    embeddings_dict = {}

    with open(filepath, 'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict



def embeddingMatrix(vocab_size, embeddings_dict, to_ix):

    embedding_matrix = np.random.uniform(size=(vocab_size, 300))
    
    for word, index in to_ix.items():
        if word in embeddings_dict.keys():
            embedding_matrix[index] = embeddings_dict[word]

    return embedding_matrix



def w2vMatrix(vocab_size, w2v, to_ix):

    embedding_matrix = np.zeros(shape=(vocab_size, 300))

    for word, index in to_ix.items():
        if word in w2v.key_to_index.keys() and word != 'null':
            embedding_matrix[index] = w2v[word]

    return embedding_matrix



def categorySelection(category, entities, attributes):

    if category == 'AIRLINE':
        y = entities[category].values
    elif category == 'SEAT':
        y = entities[category].values
    elif category == 'FOOD&DRINKS':
        y = entities[category].values
    elif category == 'ENTERTAINMENT':
        y = entities[category].values
    elif category == 'CONNECTIVITY':
        y = entities[category].values
    elif category == 'AMBIENCE':
        y = entities[category].values
    elif category == 'SERVICE':
        y = entities[category].values
    elif category == 'GENERAL':
        y = attributes[category].values
    elif category == 'PRICES':
        y = attributes[category].values
    elif category == 'QUALITY':
        y = attributes[category].values
    elif category == 'OPTIONS':
        y = attributes[category].values
    elif category == 'COMFORT':
        y = attributes[category].values
    elif category == 'WIFI':
        y = attributes[category].values
    elif category == 'SCHEDULE':
        y = attributes[category].values
    elif category == 'CABIN':
        y = attributes[category].values
    else:
        y = attributes[category].values

    return y



def leftToRight(texts, starts, ends):

    texts_left, texts_right = [], []

    for i, text in enumerate(texts):
        texts_left.append(text[:starts[i]])
        texts_right.append(text[ends[i]:])

    return texts_left, texts_right



def get_loc_info(text, start, end, aspect):

    words_left = text[:start]
    words_right = text[end + 1:]
    
    pos_info, offset_info = [], []

    if aspect != "NULL":
        for i, _ in enumerate(words_left.split()):
            pos_info.append(1 - abs(i - len(words_left))/len(text.split()))
            offset_info.append(i - len(words_left))

        for _ in range(len(aspect.split())):      
            pos_info.append(1.0)
            offset_info.append(0.0)
        
        for j, _ in enumerate(words_right.split()):
            pos_info.append(1 - (j + 1)/len(text.split()))
            offset_info.append((j + 1)/len(text.split()))
    else:
        for i, _ in enumerate(words_left.split()):
            pos_info.append(1 - abs(i - len(words_left))/len(text.split()))
            offset_info.append(i - len(words_left))

    return torch.FloatTensor(pos_info), torch.FloatTensor(offset_info)



def poSequences(texts, starts, ends, aspects):

    pos_infos, offset_infos = [], []

    for i in range(len(texts)):
        pos_info, offset_info = get_loc_info(texts[i], starts[i], ends[i], aspects[i])
        if len(pos_info) == 40:
            print(i)
        pos_infos.append(pos_info)
        offset_infos.append(offset_info)

    return pad_sequence(pos_infos, batch_first=True), pad_sequence(offset_infos, batch_first=True)