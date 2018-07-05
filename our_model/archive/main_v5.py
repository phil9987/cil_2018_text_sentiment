"""
Our model:

Currently:  Simple RNN model using Tensorflow's estimator interface.
- constant embedding
- stacked RNN (GRU or LSTM cells as basis), optional drop wrapper
- Tweet size limited to 40 words; note: all unknown words are ignored
  see stats:
  1359372 out of 1360000 tweets are <= 40 words: 99.95382352941176%
  and in test data set, only 1 of 10'000 tweets > 40 words
  5931,loool " <user> finished all the red bull . still no wings \ 355 \ 240 \ 275 \ 355 \ 270 \ 255 \ 355 \ 240 \ 275 \ 355 \ 270 \ 255 \ 355 \ 240 \ 275 \ 355 \ 270 \ 255 <url>

- adding emoj detector to have an additional classification vector (from Niko)
- adding lexicon analysis

model v5: use them at the end of analysis as separate channels

Note: some code for the lexicon analysis (loading of lexicons, counting) has already been used by
myself in the NLU project. -- Pirmin Schmid

v5.0 2018-06-19 Group The Optimists
"""

import datetime
import numpy as np
import os
import pickle
import random
import time

import tensorflow as tf

# --- configuration --------------------------------------------------------------------------------
QUICKTEST = False
VERBOSE = False

MODEL_NAME = 'v5'

IGNORE_UNKNOWN_WORDS = True
HIDDEN_STATE_SIZE = 384
SENTIMENTS = 2
RNN_STACK_DEPTH = 2
GRU = True
DROPOUT = False
DROPOUT_KEEP_PROBABILITY = 0.7

LEARNING_RATE = 1e-4
GRADIENT_CLIP = 5

TRAIN = True
EVALUATE = True
PREDICT = True

USE_RNN = True
USE_EMOJI = True
USE_LEXICON = True

BATCH_SIZE = 64
EPOCHS = 2
EVALS_PER_EPOCH = 4

BASE_DIR = '/model_checkpoints'

DATA_PATH = '../data/'

if QUICKTEST:
    BASE_DIR = '.' + BASE_DIR
    KEEP_CHECKPOINT_MAX = 1
    DIM = 25        # Dimension of embeddings. Possible choices: 25, 50, 100, 200
    TRAINING_DATA_POS = DATA_PATH + 'train_pos.txt'    # Path to positive training data
    TRAINING_DATA_NEG = DATA_PATH + 'train_neg.txt'    # Path to negative training data
    MAX_TWEET_SIZE = 30
    HIDDEN_STATE_SIZE = int(HIDDEN_STATE_SIZE / 4)
    EPOCHS = 1
else:
    user = os.getenv('USER')
    BASE_DIR = '/cluster/scratch/' + user + BASE_DIR
    KEEP_CHECKPOINT_MAX = 5  # TF default
    DIM = 200       # Dimension of embeddings. Possible choices: 25, 50, 100, 200
    TRAINING_DATA_POS = DATA_PATH + 'train_pos_full.txt'  # Path to positive training data
    TRAINING_DATA_NEG = DATA_PATH + 'train_neg_full.txt'  # Path to negative training data
    MAX_TWEET_SIZE = 30

TEST_DATA = '../data/test_data.txt'                 # Path to test data (no labels, for submission)

MODEL_NAME += '_'
MODEL_NAME += 'A' if USE_RNN else ''
MODEL_NAME += 'B' if USE_EMOJI else ''
MODEL_NAME += 'C' if USE_LEXICON else ''
if USE_RNN:
    MODEL_NAME += '_stack' + str(RNN_STACK_DEPTH)
    MODEL_NAME += '_gru' if GRU else '_lstm'
    MODEL_NAME += '_dropout' if DROPOUT else ''
    MODEL_NAME += '_size' + str(MAX_TWEET_SIZE)
    MODEL_NAME += '_dim' + str(DIM)
    MODEL_NAME += '_state' + str(HIDDEN_STATE_SIZE)
    MODEL_NAME += '_unknowns_ignored' if IGNORE_UNKNOWN_WORDS else '_with_unknowns'
MODEL_NAME += '_quicktest' if QUICKTEST else ''

MODEL_DIR = os.path.join(BASE_DIR, MODEL_NAME)

PAD = '<<pad>>'

LEXICON_BINGLIU = 'BingLiu'
LEXICON_BINGLIU_PATH = DATA_PATH + 'opinion-lexicon-English/'
LEXICON_BINGLIU_POSITIVE_FILE = LEXICON_BINGLIU_PATH + 'positive-words.txt'
LEXICON_BINGLIU_NEGATIVE_FILE = LEXICON_BINGLIU_PATH + 'negative-words.txt'

LEXICON_MPQA = 'MPQA'
LEXICON_MPQA_PATH = DATA_PATH + 'opinion-lexicon-MPQA/'
LEXICON_MPQA_FILE = LEXICON_MPQA_PATH + 'subjclueslen1-HLTEMNLP05.tff'

NEGATORS_FILE = DATA_PATH + 'negator-words.txt'

LEXICON_BOTH = LEXICON_BINGLIU + '_' + LEXICON_MPQA

run_config = tf.estimator.RunConfig(keep_checkpoint_max=KEEP_CHECKPOINT_MAX)


# --- helpers --------------------------------------------------------------------------------------
def classification_to_tf_label(classification):
    mapping = {1: 1, -1: 0}
    return mapping[classification]


def tf_label_to_classification(prediction):
    if prediction['sentiment'] > 0:
        return 1
    else:
        return -1


def evaluate_balance(text, y):
    pos = 0
    for label in y:
        if label == 1:
            pos += 1

    print('Balance of', text, 'n=', len(y), 'pos=', pos, ', ', 100.0 * pos / len(y), '%')


# --- file functions: from baseline with several modifications -------------------------------------
def load_embeddings():
    '''
        Loads the pretrained glove twitter embeddings with DIM dimensions.
        Returns a vector of embeddings and a word->idx dictionary
        which returns the index of a word in the embeddings vector
        it also returns a reverse vocabulary idx2word
    '''

    word2idx = {}  # dict to convert token to weight-idx
    idx2word = {}  # reverse vocabulary
    weights = []

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
            idx2word[index] = word
            weights.append(word_weights)
            index += 1

    return word2idx, idx2word, weights


def extend_embeddings(word2idx, idx2word, weights):
    '''Add some customized embeddings'''
    pad = [0.0] * DIM
    pad_index = len(weights)
    word2idx[PAD] = pad_index
    idx2word[pad_index] = PAD
    weights.append(pad)
    return word2idx, idx2word, weights


def read_lexicon(filename, value, lexicon=None):
    if lexicon is None:
        lexicon = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip(' \t\r\n')
            if len(l) == 0:
                continue
            if l.startswith(';'):
                continue
            lexicon[l] = value
    print('read lexicon {name}: {count} words'.format(name=filename, count=len(lexicon)))
    return lexicon


def read_mpqa_lexicon(filename, pos_lexicon=None, neg_lexicon=None):
    if pos_lexicon is None:
        pos_lexicon = {}
    if neg_lexicon is None:
        neg_lexicon = {}
    pos_count = len(pos_lexicon)
    neg_count = len(neg_lexicon)
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip(' \t\r\n')
            if len(l) == 0:
                continue
            kv_pairs = {}
            for pair in l.split(' '):
                kv = pair.split('=')
                k = kv[0]
                if len(k) == 0:
                    continue
                kv_pairs[k] = kv[1]
            w = kv_pairs.get('word1', None)
            s = kv_pairs.get('priorpolarity', None)
            if w is None or s is None:
                continue
            if s == 'positive':
                pos_lexicon[w] = 1.0
            elif s == 'negative':
                neg_lexicon[w] = 0.0
            else:
                continue
    print('read lexicon {name}: added {pos} pos, {neg} neg, {count} total words'
          .format(name=filename, pos=len(pos_lexicon)-pos_count, neg=len(neg_lexicon)-neg_count,
                        count=len(pos_lexicon)+len(neg_lexicon)-pos_count-neg_count))
    print('total in lexicon: {pos} pos, {neg} neg, {count} total words'
          .format(name=filename, pos=len(pos_lexicon), neg=len(neg_lexicon),
                  count=len(pos_lexicon)+len(neg_lexicon)))
    return pos_lexicon, neg_lexicon


MAX_EMOJI_COUNT = 9


def emoji_detector(tweet):
    '''Returns a vector of emoj counts'''
    smiles = tweet.count('<smile>')
    sadfaces = tweet.count('<sadface>')
    hearts = tweet.count('<heart>')
    neutralfaces = tweet.count('<neutralface>')
    lolfaces = tweet.count('<lolface>')
    users = tweet.count('<user>')
    hashtags = tweet.count('<hashtag>')
    elongs = tweet.count('<elong>')
    repeats = tweet.count('<repeat>')
    return [smiles, sadfaces, hearts, neutralfaces, lolfaces, users, hashtags, elongs, repeats]


def lexicon_encoding(context, tweet):
    pos = context['pos']
    neg = context['neg']
    inv = context['inv']
    p_count = 0
    n_count = 0
    i_count = 0
    t_count = 0

    for w in tweet:
        # some additional cleanup may be needed
        w = w.strip(" '\"-\\.")
        t_count += 1
        p = pos.get(w, None)
        if p is not None:
            p_count += 1
            continue
        n = neg.get(w, None)
        if n is not None:
            n_count += 1
            continue
        i = inv.get(w, None)
        if i is not None:
            i_count += 1
            continue
        # unknown
    invert = i_count % 2 == 1
    diff = p_count - n_count
    if invert:
        diff = -diff
    # currently a very short "vector" of length 1
    return [diff]


def tweet_encoding(context, tweet, word2idx, is_training):
    '''
        Returns tweet encoding by [MAX_TWEET_SIZE] integers
        unknown words are either ignored or returned as encoding for <<pad>>

        Tweets longer than MAX_TWEET_SIZE
        - during training: return None, 0 and must be ignored
        - for evaluation/test sets: return truncated data up to MAX_TWEET_SIZE

        note: the actual encoding is a lookup in tensorflow to save memory

        added vectors: emoji vector and lexicon vector
    '''
    word_list = []
    pad = word2idx[PAD]
    tokens = tweet.split()  # split tweet by whitespaces
    for word in tokens:
        i = word2idx.get(word, None)
        if i is None:
            if VERBOSE:
                print('Warning: {} is not in vocabulary...'.format(word))

            if not IGNORE_UNKNOWN_WORDS:
                word_list.append(pad)
            continue

        word_list.append(i)

    emoji_vector = emoji_detector(tweet)
    lexicon_vector = lexicon_encoding(context, tweet)

    words = len(word_list)
    if words > MAX_TWEET_SIZE:
        if is_training:
            return None, 0, emoji_vector, lexicon_vector
        else:
            return word_list[:MAX_TWEET_SIZE], MAX_TWEET_SIZE, emoji_vector, lexicon_vector

    padding = MAX_TWEET_SIZE - words
    if padding > 0:
        pads = [pad] * padding
        word_list.extend(pads)

    return word_list, words, emoji_vector, lexicon_vector


def load_trainingdata(context, word2idx):
    '''
    Loads and returns training data encoding X and correct labels y in a shuffled order
    note: each tweet is a vector [MAX_TWEET_SIZE] of integers
    actual embedding happens inside of the tensorflow model
    plus emoji and lexicon vectors
    '''
    train = []
    for tweet in open(TRAINING_DATA_POS, 'r', encoding='utf8'):
        encoding, count, emoji_vector, lexicon_vector = tweet_encoding(context, tweet, word2idx, True)
        if count == 0:
            continue
        train.append((encoding, count, emoji_vector, lexicon_vector, classification_to_tf_label(1)))

    for tweet in open(TRAINING_DATA_NEG, 'r', encoding='utf8'):
        encoding, count, emoji_vector, lexicon_vector = tweet_encoding(context, tweet, word2idx, True)
        if count == 0:
            continue
        train.append((encoding, count, emoji_vector, lexicon_vector, classification_to_tf_label(-1)))

    random.shuffle(train)  # shuffle order of training data randomly
    X, X_counts, X_emoji_vectors, X_lexicon_vectors, y = zip(*train)
    return np.asarray(X), np.asarray(X_counts), np.asarray(X_emoji_vectors, dtype=np.float32), np.asarray(X_lexicon_vectors, dtype=np.float32), np.asarray(y)


def load_testdata(context, word2idx):
    '''
    Loads and returns test data. each tweet is a vector of [MAX_TWEET_SIZE] integers
    plus emoji and lexicon vectors
    '''
    X = []
    X_counts = []
    X_emoji_vectors = []
    X_lexicon_vectors = []
    for tweet in open(TEST_DATA, 'r', encoding='utf8'):
        encoding, count, emoji_vector, lexicon_vector = tweet_encoding(context, tweet, word2idx, False)
        X.append(encoding)
        X_counts.append(count)
        X_emoji_vectors.append(emoji_vector)
        X_lexicon_vectors.append(lexicon_vector)

    # the test data is padded to align with BATCH_SIZE
    # due to a bug reported in "tensorflow ConcatOp : Dimensions of inputs should match"
    # actual_count is used to limit output of the results
    actual_count = len(X_counts)

    rem = actual_count % BATCH_SIZE
    if rem != 0:
        pad_n = BATCH_SIZE - rem
        pad_data = [X[0]] * pad_n
        pad_counts = [X_counts[0]] * pad_n
        pad_emojis = [X_emoji_vectors[0]] * pad_n
        pad_lexicon = [X_lexicon_vectors[0]] * pad_n
        X.extend(pad_data)
        X_counts.extend(pad_counts)
        X_emoji_vectors.extend(pad_emojis)
        X_lexicon_vectors.extend(pad_lexicon)

    return np.asarray(X), np.asarray(X_counts), np.asarray(X_emoji_vectors, dtype=np.float32), np.asarray(X_lexicon_vectors, dtype=np.float32), actual_count


def generate_submission(predictions, actual_count, filename):
    ''' Creates a submission file named according to filename in the current folder. '''

    with open(filename, 'w') as file:
        file.write('Id,Prediction\n')
        for i, prediction in enumerate(predictions):
            if i >= actual_count:
                break
            # additional mapping back into the desired classification space
            file.write('{},{}\n'.format(i + 1, tf_label_to_classification(prediction)))


def write_eval_data(eval_data, eval_lengths, eval_labels, predictions, eval_count, idx2word, filename):
    ''' Writes a .csv file for analysis of the eval data set. '''

    with open(filename, 'w') as file:
        file.write('Id\tPrediction\tExpected\tMatch\tText\n')
        for i, prediction in enumerate(predictions):
            if i >= eval_count:
                break
            # additional mapping back into the desired classification space
            prediction = tf_label_to_classification(prediction)
            expected = 1 if eval_labels[i] > 0 else -1
            match = prediction == expected
            text = ''
            encoding = eval_data[i]
            eval_length = eval_lengths[i]
            for j, code in enumerate(encoding):
                if j >= eval_length:
                    break
                text += idx2word[code] + ' '
            file.write('{id}\t{prediction}\t{expected}\t{match}\t{text}\n'
                       .format(id=i + 1, prediction=prediction,
                               expected=expected, match=match, text=text))


# --- RNN language model ---------------------------------------------------------------------------
def rnn_cell():
    if GRU:
        cell = tf.contrib.rnn.GRUCell(HIDDEN_STATE_SIZE)
    else:
        cell = tf.contrib.rnn.LSTMCell(HIDDEN_STATE_SIZE, forget_bias=1.0)
    if DROPOUT:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT_KEEP_PROBABILITY)
    return cell


def lang_model_fn(features, labels, mode, params):
    # lookup table for the embeddings: shape [n_embeddings, DIM]
    embeddings = tf.constant(params['embeddings'], dtype=tf.float32)

    # words: shape [BATCH_SIZE, MAX_TWEET_SIZE]
    # lengths: shape [BATCH_SIZE]
    # emojis: shape [BATCH_SIZE, MAX_EMOJI_COUNT]
    # lexicons: shape [BATCH_SIZE]
    # labels: shape [BATCH_SIZE]
    words = features['x']
    lengths = features['length']
    emojis = features['emoji']
    lexicons = features['lexicon']

    # --- A: RNN ---
    # shape [BATCH_SIZE, MAX_TWEET_SIZE, DIM]
    embedded_words = tf.nn.embedding_lookup(embeddings, words)

    # rnn_outputs: shape [BATCH_SIZE, MAX_TWEET_SIZE, HIDDEN_STATE_SIZE]
    # final_state: shape [BATCH_SIZE, HIDDEN_STATE_SIZE]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(RNN_STACK_DEPTH)])
    initial_state = stacked_rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=stacked_rnn_cell,
                                                 inputs=embedded_words,
                                                 time_major=False,
                                                 sequence_length=lengths,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32)

    # currently a one layer step from this collected output state to the 2 labels
    # can be extended to a deeper NN, of course

    # shape [BATCH_SIZE, MAX_TWEET_SIZE * HIDDEN_STATE_SIZE]
    flattened = tf.layers.flatten(inputs=rnn_outputs, name="flatten")

    # shape [BATCH_SIZE, SENTIMENTS]
    rnn_logits = tf.contrib.layers.fully_connected(inputs=flattened, num_outputs=SENTIMENTS, activation_fn=tf.sigmoid)

    # --- B: emojis ---
    emojis_logits = tf.contrib.layers.fully_connected(inputs=emojis, num_outputs=SENTIMENTS, activation_fn=tf.sigmoid)


    # --- C: lexicon ---
    lexicon_logits = tf.contrib.layers.fully_connected(inputs=lexicons, num_outputs=SENTIMENTS, activation_fn=tf.sigmoid)

    # --- combine all ---
    concat_list = []
    if USE_RNN:
        concat_list.append(rnn_logits)
    if USE_EMOJI:
        concat_list.append(emojis_logits)
    if USE_LEXICON:
        concat_list.append(lexicon_logits)
    combined = tf.concat(concat_list, 1)
    logits = tf.contrib.layers.fully_connected(inputs=combined, num_outputs=SENTIMENTS, activation_fn=tf.sigmoid)

    # shape: [BATCH_SIZE]
    sentiment_prediction = tf.argmax(logits, axis=1)

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "sentiment": sentiment_prediction
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # use identical loss function for training and eval
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    batch_loss = tf.reduce_mean(loss)

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        variables = tf.trainable_variables()
        gradients = tf.gradients(batch_loss, variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=GRADIENT_CLIP)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=batch_loss, train_op=train_op)

    # eval: accuracy
    accuracy = tf.metrics.accuracy(labels=labels, predictions=sentiment_prediction)

    metrics = {'accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=batch_loss, eval_metric_ops=metrics)


# --- main -----------------------------------------------------------------------------------------

def main():
    ''' main entry point of application '''
    tf.logging.set_verbosity(tf.logging.INFO)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
    word2idx, idx2word, embeddings = load_embeddings()
    word2idx, idx2word, embeddings = extend_embeddings(word2idx, idx2word, embeddings)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    pos_lexicon = read_lexicon(LEXICON_BINGLIU_POSITIVE_FILE, 1.0)
    neg_lexicon = read_lexicon(LEXICON_BINGLIU_NEGATIVE_FILE, 0.0)
    pos_lexicon, neg_lexicon = read_mpqa_lexicon(LEXICON_MPQA_FILE, pos_lexicon=pos_lexicon, neg_lexicon=neg_lexicon)
    inv = read_lexicon(NEGATORS_FILE, 1)

    context = {
        'pos': pos_lexicon,
        'neg': neg_lexicon,
        'inv': inv
    }

    if TRAIN or EVALUATE or PREDICT:
        X, X_lengths, X_emoji_vectors, X_lexicon_vectors, y = load_trainingdata(context, word2idx)
        n = len(X_lengths)
        eval_n = int(n / 50)        # use 2% of the data for evaluation
        rem = eval_n % BATCH_SIZE   # align to BATCH_SIZE
        diff = 0
        if rem != 0:
            diff = BATCH_SIZE - rem
        eval_n += diff
        train_n = n - eval_n
        X_train = X[:train_n]
        X_lengths_train = X_lengths[:train_n]
        X_emoji_vectors_train = X_emoji_vectors[:train_n]
        X_lexicon_vectors_train = X_lexicon_vectors[:train_n]
        y_train = y[:train_n]
        X_eval = X[train_n:]
        X_lengths_eval = X_lengths[train_n:]
        X_emoji_vectors_eval = X_emoji_vectors[train_n:]
        X_lexicon_vectors_eval = X_lexicon_vectors[train_n:]
        y_eval = y[train_n:]

        evaluate_balance('training', y_train)
        evaluate_balance('evaluation', y_eval)

    if PREDICT:
        X_test, X_lengths_test, X_emoji_vectors_test, X_lexicon_vectors_test, actual_test_count = load_testdata(context, word2idx)

    # create estimator
    params = {'embeddings': embeddings}

    sentiment_predictor = tf.estimator.Estimator(
        model_fn=lang_model_fn,
        model_dir=MODEL_DIR,
        config=run_config,
        params=params)

    if TRAIN:
        print("Training: Model", MODEL_NAME)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train, "length": X_lengths_train, "emoji": X_emoji_vectors_train, "lexicon": X_lexicon_vectors_train},
            y=y_train,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True)  # already shuffled; but still good to do more shuffling for more than 1 epoch

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_eval, "length": X_lengths_eval, "emoji": X_emoji_vectors_eval, "lexicon": X_lexicon_vectors_eval},
            y=y_eval,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=False)

        num_steps = int(n / BATCH_SIZE / EVALS_PER_EPOCH) + 1
        for i in range(EPOCHS):
            for j in range(EVALS_PER_EPOCH):
                sentiment_predictor.train(
                    input_fn=train_input_fn,
                    steps=num_steps)
                sentiment_predictor.evaluate(
                    input_fn=eval_input_fn,
                    steps=1)

    if EVALUATE:
        print("Evaluation: Model", MODEL_NAME)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_eval, "length": X_lengths_eval, "emoji": X_emoji_vectors_eval, "lexicon": X_lexicon_vectors_eval},
            y=y_eval,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=False)

        sentiment_predictor.evaluate(
            input_fn=eval_input_fn,
            steps=1)

    if PREDICT:
        print("Prediction: Model", MODEL_NAME, "for test data and create submission file")
        # create predict function
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test, "length": X_lengths_test, "emoji": X_emoji_vectors_test, "lexicon": X_lexicon_vectors_test},
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        predictions = sentiment_predictor.predict(
            input_fn=predict_input_fn)

        generate_submission(predictions, actual_test_count, 'submission_our_model_{}_{}.csv'.format(MODEL_NAME, timestamp))

        print("Prediction: Model", MODEL_NAME, "for validation data (testing and debugging)")
        # create predict function
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_eval, "length": X_lengths_eval, "emoji": X_emoji_vectors_eval, "lexicon": X_lexicon_vectors_eval},
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

        predictions = sentiment_predictor.predict(
            input_fn=predict_input_fn)

        write_eval_data(X_eval, X_lengths_eval, y_eval, predictions, eval_n, idx2word, 'eval_our_model_{}_{}.csv'.format(MODEL_NAME, timestamp))


if __name__ == '__main__':
    main()
