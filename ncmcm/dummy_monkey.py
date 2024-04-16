import scipy
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ncmcm.classes import *
from IPython.display import display
import os
import pickle
import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ncmcm.classes import *
from IPython.display import display
import os
import pickle
import time

os.chdir('..')
os.chdir('ncmcm')
print(os.getcwd())

''' 
This is data from the monkey brain.
'''

# Now you can import the script directly
filepath = "/Users/michaelhofer/Downloads/s1-kinematics/reaching_experiments/"
data_dict = scipy.io.loadmat(filepath + 'C_20170912_COactpas_TD.mat')

for i in data_dict['trial_data']:
    for data in i:
        for idx, arrays in enumerate(data):
            print(f'INDEX: {idx}')
            #print(np.unique(arrays))
            print(arrays.shape)
            print(len(arrays[0]))
    #print(data_dict[i])

data_dict['trial_data'][0][0][5]


X = np.zeros((188347, 171))
# current_idx = 0
# for neurons in range(8):
#     print(data_dict['trial_data'][0][0][15+neurons].shape)
#     new_rows = data_dict['trial_data'][0][0][15+neurons].shape[1]
#     current_idx += new_rows
#     print(current_idx, current_idx + new_rows)
#     print(X[:, current_idx:current_idx + new_rows].shape)
#     X[:, current_idx:current_idx + new_rows] = data_dict['trial_data'][0][0][15+neurons]
#
# X[np.isnan(X)] = 0
# print(type(X.T[3, 0]))
# print((X.T[3, 0]))
