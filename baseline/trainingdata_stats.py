import os
import numpy as np
import random
import pickle
import time
import datetime
import matplotlib.pyplot as plt

# --- settings to test for ---
DIM = 200                       # Dimension of embeddings. Possible choices: 25, 50, 100, 200

TRAINING_DATA_POS = '../data/train_pos_full.txt'    # Path to positive training data
TRAINING_DATA_NEG = '../data/train_neg_full.txt'    # Path to negative training data
TEST_DATA = '../data/test_data.txt'                 # Path to test data (no labels, for submission)


def load_word_indices():
    ''' Just the index dictionary '''

    word2idx = {}  # dict to convert token to weight-idx

    with open('../data/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(DIM), 'r') as file:
        index = 0
        for line in file:
            values = line.split()  # word and weights are separated by space
            word = values[0]  # word is first element on each line
            word_weights = values[1:]
            if len(word_weights) != DIM:
                print('wrong encoding length {} for ""; word ignored'.format(len(word_weights), word))
                continue

            word2idx[word] = index
            index += 1

    return word2idx


def load_trainingdata():
    ''' list of tweets'''

    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        train.append(tweet)

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        train.append(tweet)

    return train


def load_testdata():
    ''' Loads and returns training data embeddings X and correct labels y in a randomized order '''

    test = []
    for tweet in open(TEST_DATA, 'r', encoding='utf8'):
        test.append(tweet)

    return test


def number_words(sentence, word2idx):
    words = sentence.split(' ')
    n_words = len(words)
    n_known_words = 0
    for w in words:
        i = word2idx.get(w, None)
        if i is None:
            continue
        n_known_words += 1
    return n_words, n_known_words


def print_missing(text, all, known, cutoff):
    missing_all = len([x for x in all if x > cutoff])
    missing_known = len([x for x in known if x > cutoff])
    print('Cutoff', cutoff, 'missing in', text, ': for length', missing_all, ',', 100.0 * missing_all / len(all),
          '%, for known words', missing_known, ',', 100.0 * missing_known / len(known), '%')


def main():
    word2idx = load_word_indices()
    train = load_trainingdata()
    training_tweet_lengths = [number_words(tweet, word2idx) for tweet in train]
    training_word_counts, training_known_word_counts = zip(*training_tweet_lengths)

    test = load_testdata()
    test_tweet_lengths = [number_words(tweet, word2idx) for tweet in test]
    test_word_counts, test_known_word_counts = zip(*test_tweet_lengths)

    # plt.subplot(221)
    plt.hist(training_word_counts, bins='auto')
    plt.xlim([0, 60])
    # plt.title('Training: all')
    plt.xlabel('words per tweet')
    plt.ylabel('tweets')
    # plt.subplot(222)
    # plt.hist(training_known_word_counts, bins='auto')
    # plt.xlim([0, 60])
    # plt.title('Training: known')
    # plt.xlabel('words per tweet')
    # plt.ylabel('tweets')
    # # plt.suptitle('Training')c

    # plt.subplot(223)
    # plt.hist(test_word_counts, bins='auto')
    # plt.xlim([0, 60])
    # plt.title('Test: all')
    # plt.xlabel('words per tweet')
    # plt.ylabel('tweets')
    # plt.subplot(224)
    # plt.hist(test_known_word_counts, bins='auto')
    # plt.xlim([0, 60])
    # plt.title('Test: known')
    # plt.xlabel('words per tweet')
    # plt.ylabel('tweets')
    # # plt.suptitle('Counts')

    plt.tight_layout()

    for limit in [20, 30, 40]:
        print_missing('training', training_word_counts, training_known_word_counts, limit)
        print_missing('test', test_word_counts, test_known_word_counts, limit)

    plt.show()


if __name__ == '__main__':
    main()