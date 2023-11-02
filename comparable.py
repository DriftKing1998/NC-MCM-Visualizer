from collections import Counter
from classes import *
import random

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
data = Database(worm_num, verbose=0)
data.exclude_neurons(b_neurons)
loaded_vs = data.loadBundleVisualizer()

loaded_vs.plot3D_mapping(show_legend=True, quivers=True)

weights_T_Y = loaded_vs.model.T_Y.get_weights()
weights_predictor = loaded_vs.model.predictor.get_weights()


for worm_num in range(5):
    print(f'Worm number #{worm_num}')
    data = Database(worm_num, verbose=0)
    data.exclude_neurons(b_neurons)
    vs = data.loadBundleVisualizer()

    ### Set weights of T_Y and predictor
    vs.model.T_Y.set_weights(weights_T_Y)
    vs.model.predictor.set_weights(weights_predictor)

    ### Freeze weights of predictor and T_Y
    vs.model.T_Y.trainable = False
    vs.model.predictor.trainable = False

    vs.train_model(epochs=500)
    vs.plot_loss()
    vs.plot3D_mapping(show_legend=True)
exit()

# This follows worm 0 looks best.
'''
for worm_num in range(5):
    data = Database(worm_num, verbose=0)
    data.exclude_neurons(b_neurons)
    loaded_vs = data.loadBundleVisualizer()
    loaded_vs.plot3D_mapping(show_legend=True)
'''