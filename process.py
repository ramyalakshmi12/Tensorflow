import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
import io
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000000
loxi = None

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with io.open(fi, 'r', encoding='cp437') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += all_words
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    #w_counts = {'the':79797, 'and': 89874}
    l2 = []
    for w in w_counts:
        if 70000 > w_counts[w] > 10:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with io.open(sample, 'r', encoding='cp437') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    global loxi
    loxi = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', loxi, [1, 0])
    features += sample_handling('neg.txt', loxi, [0, 1])
    
    random.shuffle(features)
    features = np.array(features)
    
    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('lexicon.pickle', 'wb') as f:
        pickle.dump(loxi, f)

