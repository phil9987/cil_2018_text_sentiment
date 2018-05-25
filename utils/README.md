# Leonhard Cluster
The cluster is especially useful for the training of the neural networks.
login via ssh:
ssh <user>@login.leonhard.ethz.ch


# Setup (one time)
. install_on_leonhard.source
download / transfer training and test data to leonhard
download / transfer glove embeddings to leonhard
useful scp commands (transfers local file to remote location):
scp path/to/local/file <user>@login.leonhard.ethz.ch:path/to/remote/file


# Preparation (for each session)
. activate_on_Leonhard.source

# Submit Job
bsub -B -N -n 4 -R "rusage[mem=16000,ngpus_excl_p=1]" python3 simple_rnn.py
