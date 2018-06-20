import os
import pickle
import time
import datetime
import operator
import random
import numpy as np
import nltk.sentiment.util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

''' global configuration '''

EMBEDDING_DIMENSIONS = 200 # one of {25 50 100 200}
TRAINING_DATA_POS = '../data/train_pos.txt'
TRAINING_DATA_NEG = '../data/train_neg.txt'
TEST_DATA = '../data/test_data.txt' # no labels, for submission
VERBOSE_LOGGING = False
UNKNOWN_WORDS = {}

# Random forest classifier parameters
MAX_DEPTH = None
N_ESTIMATORS = 20

def load_embeddings():
    """Loads the pretrained glove twitter embeddings with the number of
    dimensions specified in EMBEDDING_DIMENSIONS. Returns a vector of
    embeddings and a word->idx dictionary which maps words to their
    indices in the embeddings vector.
    """

    word2idx = {} # dict to convert token to weight-idx
    weights = []

    with open('../data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(EMBEDDING_DIMENSIONS), 'r') as file:
        for index, line in enumerate(file):
            values = line.split()       # word and weights are separated by space
            word = values[0]            # word is first element on each line
            word_weights = np.asarray(values[1:], dtype=np.float32) # weights for current word
            word2idx[word] = index
            weights.append(word_weights)

    return word2idx, weights
 
def tweet_embedding(tweet, word2idx, embeddings):
    """Computes tweet embedding by averaging embeddings of all words
    contained in it. Words which are not contained in vocabulary are
    ignored.
    """

    global UNKNOWN_WORDS
    
    vec_list = []
    tokens = tweet.split()

    tokens = nltk.sentiment.util.mark_negation(tokens)
    
    for word in tokens:
        try:
            i = None
            neg_idx = word.rfind('_NEG')
            if neg_idx > 0:
                word = word[:neg_idx]
                i = np.negative(word2idx[word])
            else:
                i = word2idx[word]
            
            vec_list.append(embeddings[i])
            
        except KeyError:
            # ignore the word if it's not in vocabulary
            if VERBOSE_LOGGING:
                UNKNOWN_WORDS[word] = UNKNOWN_WORDS.get(word, 0) + 1
                print('Warning: ignoring {} as it is not in vocabulary...'.format(word))
            pass

    if len(vec_list) == 0:
        if VERBOSE_LOGGING:
            print('Warning: zero length tweet: {}'.format(tweet))
        return np.zeros(EMBEDDING_DIMENSIONS)

    return np.mean(vec_list, axis=0)

def load_trainingdata(word2idx, embeddings):
    """Loads and returns training data embeddings X and correct labels y
    in a randomized order.
    """
    
    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), 1))

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        train.append((tweet_embedding(tweet, word2idx, embeddings), -1))
        
    random.shuffle(train)

    # return embeddings and labels as two separate vectors
    return zip(*train)

def load_testdata(word2idx, embeddings):
    """Loads and returns test data as a vector of embeddings."""
    
    test = []
    for tweet in open(TEST_DATA, 'r', encoding='utf8'):
        test.append(tweet_embedding(tweet, word2idx, embeddings))
        
    return test

def generate_submission(labels, filename):
    """Creates a submission file named according to filename in the
    current folder.
    """
    
    with open(filename, 'w') as file:
        file.write('Id,Prediction\n')
        for i, label in enumerate(labels):
            file.write('{},{}\n'.format(i+1, label))

def main():
    global UNKNOWN_WORDS
    
    timestamp = datetime.datetime \
                        .fromtimestamp(time.time()) \
                        .strftime('%Y-%m-%d_%H%M%S')
    
    word2idx, embeddings = load_embeddings()
    
    X, y = load_trainingdata(word2idx, embeddings)

    clf = RandomForestClassifier(max_depth = MAX_DEPTH,
                                 n_estimators = N_ESTIMATORS,
                                 random_state = 0,
                                 n_jobs = -1)

    # calculate cross validation score using 5 splits
    scores = cross_val_score(clf, X, y, cv=5)
    print('cross validation scores calculated ({}-dimensional embeddings, max_depth={})'.format(EMBEDDING_DIMENSIONS, MAX_DEPTH))
    print(scores)

    # train classifier on whole dataset
    clf.fit(X, y)

    # store trained classifier
    classifier_file = 'random_forest_classifier_MaxDepth{}_DIM{}_{}'.format(MAX_DEPTH, EMBEDDING_DIMENSIONS, timestamp)
    pickle.dump(clf, open(classifier_file, 'wb'))
    
    testdata = load_testdata(word2idx, embeddings)
    if VERBOSE_LOGGING:
        print("total number of distinct unknown words: {}.".format(len(UNKNOWN_WORDS)))
        
    with open("unkown_words.txt", 'w') as file:
        for word in UNKNOWN_WORDS.keys():
            print(word, file=file)

    # predict labels of test data
    submission_labels = clf.predict(testdata)
    submission_file = 'submission_randomForest_MaxDepth{}_DIM{}_{}.csv'.format(MAX_DEPTH, EMBEDDING_DIMENSIONS, timestamp)
    generate_submission(submission_labels, submission_file)

if __name__ == '__main__':
    main()
