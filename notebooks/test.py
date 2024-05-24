from ncmcm import *
from IPython.display import display
import os
import pickle

os.chdir('..')
print(os.getcwd())
# For loading the data
num = 4
with open(f'/Users/michaelhofer/Documents/GitHub/ncmcm/ncmcm/data/pickles/data_worm_{num}.pkl', 'rb') as file:
	data = pickle.load(file)
data.name = f'worm {num}'
vs = data.createVisualizer(epochs=2500)

vs.make_comparison(show_legend=True)