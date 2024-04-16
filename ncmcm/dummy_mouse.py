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
This is data from the mouse brain.
'''

# Now you can import the script directly
filepath = "/Users/michaelhofer/Documents/Uni/Mouse_data/mouse_10/sub-10_ses-mouse-10-session-date-2017-09-21-area-V1-L23-multi-plane-imaging_behavior+ophys.nwb"
# Open the file in read mode "r", and specify the driver as "ros3" for S3 files
io = NWBHDF5IO(filepath, mode="r", load_namespaces=True)
nwbfile = io.read()

deconv_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].data[:]
imaging_timestamps_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].timestamps

print('Shape of neuronal data for each imaging Plane')
print(deconv_0.T.shape) # Neurons | Timesteps
proc = nwbfile.processing

print(proc['behavior']['frame_aligned_position']['frame_aligned_forward_and_lateral_position'])
print(imaging_timestamps_0.shape)
bhv = proc['behavior']
time_diffs_0 = np.diff(imaging_timestamps_0)
mean_time_step_0 = np.mean(time_diffs_0)
fps_0 = 1 / mean_time_step_0
print(f'FPS {fps_0}')

pos = bhv['frame_aligned_position']['frame_aligned_forward_and_lateral_position']

# Assuming pos.data is a NumPy array with shape (188740, 2)
# and imaging_timestamps is a NumPy array with shape (37748,)
forward_threshold = 0.001
lateral_threshold = 0.001

# Initialize an empty array to store behaviors
movements = np.zeros(imaging_timestamps_0.shape)

# Loop through imaging timestamps and categorize behaviors
for i, timestamp in enumerate(imaging_timestamps_0):
    print(f'TIMESTAMP: {timestamp}')
    # Calculate the index directly from the pattern (every 5th index)
    index = i * 5

    # Get the forward and lateral positions at the current index
    forward_pos, lateral_pos = pos.data[index]

    # Get the forward and lateral positions at the previous index
    prev_forward_pos, prev_lateral_pos = pos.data[index - 1]

    # Calculate changes in forward and lateral positions
    forward_change = forward_pos - prev_forward_pos
    lateral_change = lateral_pos - prev_lateral_pos

    # Categorize behaviors based on changes
    if np.abs(forward_change) < forward_threshold and np.abs(lateral_change) < lateral_threshold:
        movements[i] = 0
    elif forward_change > forward_threshold:
        if lateral_change > lateral_threshold:
            movements[i] = 2
        elif lateral_change < -lateral_threshold:
            movements[i] = 3
        else:
            movements[i] = 1
    elif forward_change < -forward_threshold:
        if lateral_change > lateral_threshold:
            movements[i] = 5
        elif lateral_change < -lateral_threshold:
            movements[i] = 6
        else:
            movements[i] = 4

    elif lateral_change > lateral_threshold:
        movements[i] = 7
    elif lateral_change < -lateral_threshold:
        movements[i] = 8
    else:
        movements[i] = 9

# Now, the 'behaviors' array contains the categorized behaviors for each frame in imaging timestamps.
print(movements.shape)
states = ['standing still',
          'moving forward', 'moving forward and right', 'moving forward and left',
          'moving backward', 'moving backward and right', 'moving backward and left',
          'going right', 'going left', 'invisible']
print(np.unique(movements, return_counts=True))
mouse_plane0_complete = Database(neuron_traces=deconv_0.T,
                                 behavior=movements,
                                 behavioral_states=states,
                                 fps=fps_0)
mouse_plane0_complete.plotting_neuronal_behavioral()

logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
mouse_plane0_complete.fit_model(logreg, ensemble=True)
mouse_plane0_complete.cluster_BPT(nrep=10, max_clusters=10)

os.chdir('/Users/michaelhofer/Documents/Uni/THESIS/Plots')

go_on = 1
while go_on:
    num = int(input('Amount of clusters: '))
    mouse_plane0_complete.step_plot(clusters=num)
    go_on = int(input('Go on (0/1): '))

go_on = 1
while go_on:
    num = int(input('Amount of clusters: '))
    mouse_plane0_complete.behavioral_state_diagram(cog_stat_num=num, interactive=True)
    go_on = int(input('Go on (0/1): '))

go_on = 1
while go_on:
    num = int(input('Amount of epochs: '))
    mouse_plane0_vs = mouse_plane0_complete.createVisualizer(epochs=num)
    mouse_plane0_vs.plot_mapping(show_legend=True, quivers=True)
    mouse_plane0_vs.make_comparison(show_legend=True, quivers=True)
    go_on = int(input('Go on (0/1): '))
