import pickle
from scripts.classes import *

# Load the pickled data back into a Python object
data = []
for i in range(5):
    with open(f'data/pickles/data_worm_{i + 1}_8D.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
        #loaded_data.behavioral_state_diagram(interactive=True, cog_stat_num=1)#, threshold=0.001)
        loaded_data.step_plot(clusters=3)
        #data.append(loaded_data.p_markov)


#data = np.asarray(data)
#data = data.mean(axis=2)
#average_markov_plot(data)
