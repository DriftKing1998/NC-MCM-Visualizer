from scripts.classes import *
from IPython.display import display
import os

os.chdir('..')
print(os.getcwd())

# Assuming 'A' and 'B' are encoded as strings in your 'Y' variable
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL'
]

worm_num = 0

data = Database(worm_num, verbose=1)
data.exclude_neurons(b_neurons)

# Adding prediction Model & Cluster BPT
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg, binary=True)

data.cluster_BPT(nrep=3, max_clusters=20, plot_markov=True)
