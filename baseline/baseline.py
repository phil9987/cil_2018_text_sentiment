import os
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
import datetime

''' global configuration '''
DIM = 200       # Dimension of embeddings. Possible choices: 25, 50, 100, 200
TRAINING_DATA_POS = '../data/train_pos_full.txt'    # Path to positive training data
TRAINING_DATA_NEG = '../data/train_neg_full.txt'    # Path to negative training data
TEST_DATA = '../data/test_data.txt'                 # Path to test data (no labels, for submission)
VERBOSE = False                                     # Want lots of info on terminal?
MAX_DEPTH = None                                    # Max-depth for RandomForest Classifier
N_ESTIMATORS = 20

def load_embeddings():
    '''
        Loads the pretrained glove twitter embeddings with DIM dimensions.
        Returns a vector of embeddings and a word->idx dictionary
        which returns the index of a word in the embeddings vector
    '''

    word2idx = { } # dict to convert token to weight-idx
    weights = []

    with open('../data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(DIM), 'r') as file:
        for index, line in enumerate(file):
            values = line.split()       # word and weights are separated by space
            word = values[0]            # word is first element on each line
            word_weights = np.asarray(values[1:], dtype=np.float32) # weights for current word
            word2idx[word] = index
            weights.append(word_weights)

    return word2idx, weights
 
def tweet_embedding(tweet, word2idx, embeddings):
    '''
        Returns tweet embedding by averaging embeddings of all contained words
        Words which are not contained in vocabulary are ignored
    '''

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
    ''' Loads and returns training data embeddings X and correct labels y in a randomized order '''

    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), 1))

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), -1))
    
    random.shuffle(train)       # shuffle order of training data randomly
    return zip(*train)          # return embeddings and labels as two separate vectors

def load_testdata(word2idx, embeddings):
    ''' Loads and returns test data as a vector of embeddings '''

    test = []
    for line in open(TEST_DATA, 'r', encoding='utf8'):
        split_idx = line.find(',')  # first occurrence of ',' is separator between id and tweet
        tweet = line[(split_idx + 1):]
        test.append(tweet_embedding(tweet, word2idx, embeddings))
    return test

def generate_submission(labels, filename):
    ''' Creates a submission file named according to filename in the current folder. '''

    with open(filename, 'w') as file:
        file.write('Id,Prediction\n')
        for i, label in enumerate(labels):
            file.write('{},{}\n'.format(i+1, label))


def main():
    ''' main entry point of application '''

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
    word2idx, embeddings = load_embeddings()
    X, y = load_trainingdata(word2idx, embeddings)
    pickle.dump( word2idx, open('word2idx_{}.pkl'.format(timestamp), 'wb'))
    pickle.dump( embeddings, open('embeddings_{}.pkl'.format(timestamp), 'wb'))
    clf = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=0, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5)                   # calculate cross validation score using 5 splits
    print('cross validation scores calculated ({}-dimensional embeddings, max_depth={})'.format(DIM, MAX_DEPTH))
    print(scores)
    clf.fit(X, y)                                               # train classifier on whole dataset
    pickle.dump( clf, open('random_forest_classifier_MaxDepth{}_DIM{}_{}'.format(MAX_DEPTH, DIM, timestamp) ,'wb'))   # store trained classifier
    testdata = load_testdata(word2idx, embeddings)
    submission_labels = clf.predict(testdata)                   # predict labels of test data
    generate_submission(submission_labels, 'submission_randomForest_MaxDepth{}_DIM{}_{}.csv'.format(MAX_DEPTH, DIM, timestamp))

if __name__ == '__main__':
    main()