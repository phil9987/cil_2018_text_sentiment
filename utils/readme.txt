# Leonhard Cluster
login via ssh:
ssh <user>@login.leonhard.ethz.ch


# Setup (one time)
. install_on_leonhard.source

# Preparation (for each session)
. activate_on_Leonhard.source

# Submit Job
bsub -B -N -n 4 -R "rusage[mem=16000,ngpus_excl_p=1]" python3 simple_rnn.py