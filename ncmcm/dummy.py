from ncmcm.classes import *
from IPython.display import display
import os
import pickle
os.chdir('..')
os.chdir('ncmcm')
print(os.getcwd())

data_average_markov = []
WORMS = []
for i in range(5):
    with open(f'data/pickles/data_worm_{i}.pkl', 'rb') as file:
        worm = pickle.load(file)
        WORMS.append(worm)
        print(worm.p_memoryless.shape)
        data_average_markov.append(np.mean(worm.p_memoryless, axis=1))

#average_markov_plot(np.asarray(data_average_markov))

# Do some cool plots

data = WORMS[0]
data.behavioral_state_diagram(save=True, show=False, adj_matrix=True)
exit()
vs1 = data.createVisualizer()
vs1.plot_mapping()

data_small = vs1.use_mapping_as_input()
logreg = LogisticRegression()
data_small.fit_model(logreg)

data_small.cluster_BPT(nrep=10, max_clusters=15)