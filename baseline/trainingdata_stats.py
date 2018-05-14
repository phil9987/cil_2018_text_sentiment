import os
import numpy as np
import random
import pickle
import time
import datetime
import matplotlib.pyplot as plt

DIM = 200       # Dimension of embeddings. Possible choices: 25, 50, 100, 200
TRAINING_DATA_POS = '../data/train_pos.txt'    # Path to positive training data
TRAINING_DATA_NEG = '../data/train_neg_full.txt'    # Path to negative training data
TEST_DATA = '../data/test_data.txt'                 # Path to test data (no labels, for submission)


def load_trainingdata():
    ''' Loads and returns training data embeddings X and correct labels y in a randomized order '''

    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        train.append(tweet)

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        train.append(tweet)

    for tweet in open(TEST_DATA, 'r', encoding='utf8'):
        train.append(tweet)
    
    return train

def number_words(sentence):
    return len(sentence.split(' '))


def main():
    train = load_trainingdata()
    tweet_lengths = [number_words(tweet) for tweet in train]
    plt.hist(tweet_lengths, bins='auto')
    plt.title('Histogram of tweet lengths')
    smaller_40 = [x for x in tweet_lengths if x <=40]
    print('{} out of {} tweets are <= 40 words: {}%'.format(len(smaller_40), len(tweet_lengths), len(smaller_40)*100/len(tweet_lengths)))

    test_larger_40 = 0
    test_size = 0
    for tweet in open(TEST_DATA, 'r', encoding='utf8'):
        test_size += 1
        if number_words(tweet) > 40:
            test_larger_40 += 1
            print(tweet)
            print(number_words(tweet))
    print('{} of {} tweets in the test data set are > 40 words'.format(test_larger_40, test_size))

    plt.show()

if __name__ == '__main__':
    main()