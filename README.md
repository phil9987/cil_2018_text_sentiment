# Project for Computational Intelligence Lab at ETH Zurich, Spring 2018

The goal of this project is to build a sentiment classifier that
predicts whether a tweet text used to include a positive smiley `:)`
or a negative smiley `:(`, based on the remaining text.

## One-time Setup on Leonhard Cluster

The cluster is especially useful for the training of the neural
networks. Login via ssh:

``` shell
ssh <user>@login.leonhard.ethz.ch
```

Copy the script at `utils/install_on_leonhard.source` to your home
dirctory (`scp ./utils/install_on_leonhard.source
<user>@login.leonhard.ethz.ch`)and source it. Then clone this
repository.

## Downloading Data

Navigate into the `data/` sub-directory and download the datasets via:

``` shell
curl http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip -o twitter-datasets.zip
unzip twitter-datasets.zip
mv twitter-datasets/* .
```

Then download the pre-trained word embeddings from the GloVe project:

``` shell
curl https://nlp.stanford.edu/data/glove.twitter.27B.zip -o glove.twitter.27B.zip
unzip glove.twitter.27B.zip -d glove.twitter.27B
```

This should leave the data directory in the following state:

``` shell
data/test_data.txt
data/train_neg_full.txt
data/train_neg.txt
data/train_pos_full.txt
data/train_pos.txt

data/glove.twitter.27B/glove.twitter.27B.25d.txt
data/glove.twitter.27B/glove.twitter.27B.50d.txt
data/glove.twitter.27B/glove.twitter.27B.100d.txt
data/glove.twitter.27B/glove.twitter.27B.200d.txt
```

## Preparing Data

From inside the `utils/` sub-directory run the pre-processing script:

``` shell
cd utils
source preprocess_data.source
```

## Preparation (for each session)

From inside the `utils/` sub-directory run the activation script:

``` shell
cd utils
source activate_on_leonhard.source
```

## Submitting a Job

``` shell
bsub -B -N -n 4 -R "rusage[mem=16000,ngpus_excl_p=1]" python3 simple_rnn.py
```

## Baseline I: Random Forest Classifier

## Baseline II: Recurrent Neural Network with LSTM

## Our Model

Recurrent neural network with additions and tweaks to make it perform
better.
