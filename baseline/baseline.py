import os
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
import datetime

DIM = 100
TRAINING_DATA_POS = '../data/train_pos_full.txt'
TRAINING_DATA_NEG = '../data/train_neg_full.txt'
TEST_DATA = '../data/test_data.txt'
VERBOSE = False
MAX_DEPTH = 10

def load_embeddings():
    word2idx = { } # dict to convert token to weight-idx
    weights = []

    with open('../data/glove.twitter.27B/glove.twitter.27B.100d.txt', 'r') as file:
        for index, line in enumerate(file):
            values = line.split()       # word and weights are separated by space
            word = values[0]            # word is first element on each line
            word_weights = np.asarray(values[1:], dtype=np.float32) # weights for current word
            word2idx[word] = index
            weights.append(word_weights)
            
            #if index + 1 == 40000:     # for now only take 40k most frequent words
            #    break

    return word2idx, weights


# generate tweet embedding by averaging embeddings of all containted words
def tweet_embedding(tweet, word2idx, embeddings):
    vec_list = []
    tokens = tweet.split()      # split tweet by whitespaces

    for word in tokens:
        try:
            i = word2idx[word]
            vec_list.append(embeddings[i])
        except KeyError:
            # Ignore a word if it's not in vocabulary
            if VERBOSE:
                print('Warning: ignoring {} as it is not in vocabulary...'.format(word))
            pass

    if len(vec_list) == 0:
        if VERBOSE:
            print('Warning: zero length tweet: {}'.format(tweet))
        return np.zeros(DIM)

    return np.mean(vec_list, axis=0)

def load_trainingdata(word2idx, embeddings):
    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), 1))

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), -1))
    
    random.shuffle(train)   # shuffle order of training data randomly

    return train

def load_testdata(word2idx, embeddings):
    test = []
    for line in open(TEST_DATA, 'r', encoding='utf8'):
        split_idx = line.find(',')  # first occurrence of ',' is separator between id and tweet
        tweet = line[(split_idx + 1):]
        test.append(tweet_embedding(tweet, word2idx, embeddings))
    return test

def generate_submission(labels, filename):
    
    with open(filename, 'w') as file:
        file.write('Id,Prediction\n')
        for i, label in enumerate(labels):
            file.write('{},{}\n'.format(i+1, label))


def main():
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    word2idx, embeddings = load_embeddings()
    # print(tweet_embedding("<user> i could actually kill that girl i'm so sorry ! ! !", word2idx, embeddings))
    training_data = load_trainingdata(word2idx, embeddings)
    pickle.dump( word2idx, open('word2idx.pkl', 'wb'))
    pickle.dump( embeddings, open('embeddings.pkl', 'wb'))
    X, y = zip(*training_data)
    clf = RandomForestClassifier(max_depth=MAX_DEPTH, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    print('cross validation scores calculated ({}-dimensional embeddings, max_depth={})'.format(DIM, MAX_DEPTH))
    print(scores)
    clf.fit(X, y)
    pickle.dump( clf, open('random_forest_classifier' ,'wb'))
    # clf = pickle.load(open('random_forest_classifier', 'rb'))
    testdata = load_testdata(word2idx, embeddings)
    submission_labels = clf.predict(testdata)
    generate_submission(submission_labels, 'submission_{}.csv'.format(timestamp))

if __name__ == '__main__':
    main()