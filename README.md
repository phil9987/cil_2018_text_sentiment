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

The code for the random forest classification is contained in the
`baseline/` sub-directory. It can be run via `python baseline.py`. On
the cluster, a job should be started as follows:

``` shell
module load python
bsub -n 1 -R "rusage[mem=16384]" -W 3:59 "python baseline.py"
```

It's worth to have a look at the documentation of scikit learn, there
are many parameters which can be explored:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## Baseline II: Recurrent Neural Network with LSTM

The code for the recurrent neural net baseline is contained in the
`baseline_simple_nn/` sub-directory.

## Our Model

The code for our own model is contained in the `our_model/`
sub-directory. Our approach combines a recurrent neural network with
additions and tweaks to make it perform better. We utilize
TensorFlow's estimator interface.
