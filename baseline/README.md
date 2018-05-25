# What is this?
This is a baseline based on a random forest classifier.

# How can I run it?
python baseline.py

# How can I further improve it?
It's worth to have a look at the documentation of scikit learn, there are many parameters which can be explored:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# How can I run it on the euler cluster?
Enter the following commands:
module load python
bsub -n 1 -R "rusage[mem=4096]" -W 3:59 "python baseline.py"

