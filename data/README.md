# Data

Inside the ETH network, navigate into this directory and download the
datasets via:

``` shell
curl http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip -o twitter-datasets.zip
unzip twitter-datasets.zip
```

Then download pre-trained word embeddings from the GloVe project:

``` shell
curl https://nlp.stanford.edu/data/glove.twitter.27B.zip -o glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

This should leave the data directory in the following state:

``` shell
data/test_data.txt
data/train_neg_full.txt
data/train_neg.txt
data/train_pos_full.txt
data/train_pos.txt
data/glove.twitter.27B/glove.twitter.27B.25d.txt
data/glove.twitter.27B/glove.twitter.27B.200d.txt
```
