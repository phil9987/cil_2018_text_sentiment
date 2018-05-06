import os
import numpy as np

DIM = 25

def load_embeddings():
    word2idx = { } # dict to convert token to weight-idx
    weights = []

    with open('../data/glove.twitter.27B/glove.twitter.27B.25d.txt', 'r') as file:
        for index, line in enumerate(file):
            values = line.split()       # word and weights are separated by space
            word = values[0]            # word is first element on each line
            word_weights = np.asarray(values[1:], dtype=np.float32) # weights for current word
            word2idx[word] = index
            weights.append(word_weights)
            
            if index + 1 == 40000:     # for now only take 40k most frequent words TODO: remove this for training later on
                break

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
            print('Warning: ignoring %s as it is not in vocabulary...' % word)
            pass

    if len(vec_list) == 0:
        print('Warning: zero length tweet: %s' % tweet)
        return np.zeros(DIM)

    return np.mean(vec_list, axis=0)


def main():
    word2idx, embeddings = load_embeddings()
    print(tweet_embedding("<user> i could actually kill that girl i'm so sorry ! ! !", word2idx, embeddings))

if __name__ == '__main__':
    main()