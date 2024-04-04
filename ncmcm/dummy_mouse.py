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


worm_num = 3
matlab = Loader(worm_num)
data = Database(*matlab.data, name='Worm 3')
#lg = LogisticRegression(max_iter=1000)
#data.fit_model(lg, ensemble=True)
data.step_plot()
vs = data.createVisualizer(epochs=2000)
vs.make_comparison()
exit()
data.cluster_BPT_single(nclusters=2, nrep=30, chunks=None)
data.behavioral_state_diagram(cog_stat_num=2, interactive=True, adj_matrix=True)
exit()


''' 
This is data from the mouse brain.
'''

# Now you can import the script directly
filepath = "/Users/michaelhofer/Documents/Uni/Mouse_data/mouse_10/sub-10_ses-mouse-10-session-date-2017-09-21-area-V1-L23-multi-plane-imaging_behavior+ophys.nwb"
# Open the file in read mode "r", and specify the driver as "ros3" for S3 files
io = NWBHDF5IO(filepath, mode="r", load_namespaces=True)
nwbfile = io.read()
#print(nwbfile.processing['ophys']['ImageSegmentation'])
print(nwbfile)
bhv = nwbfile.processing['behavior']
different_data = list(bhv.data_interfaces.keys())
t = bhv.data_interfaces['frame_aligned_position'].spatial_series['frame_aligned_forward_and_lateral_position'].data
pos = bhv['frame_aligned_position']['frame_aligned_forward_and_lateral_position']
vel = bhv['frame_to_verm_index_conversion']
start = pos.data[2458,:]
time = pos.timestamps

# extract data from deconvolved_activity_plane_0 and
imaging_timestamps_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].timestamps
print(np.asarray(imaging_timestamps_0)[:15])

deconv_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].data[:]
df_0 = nwbfile.processing['ophys']['df_over_f_plane_0']['dF_over_F_plane_0'].data[:]
x = nwbfile.processing['ophys']['df_over_f_plane_0']['dF_over_F_plane_0']
print(type(x))
print(imaging_timestamps_0.shape, deconv_0.shape, df_0.shape)
print('\nChange in fluorescence (df/F) on plane 0 for each neuron (only 7 neurons & 3 timesteps shown)')
print(pd.DataFrame(df_0).iloc[:, 0:7].head(3))
print('\nDeconvolved neural activity (actual firing) on plane 0 for each neuron (only 7 neurons & 3 timesteps shown)')
print(pd.DataFrame(deconv_0).iloc[:, 0:7].head(3))
print('\nTimestamp of each recording (only 3 timesteps shown)')
print(pd.DataFrame(imaging_timestamps_0).head(3))


deconv_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].data[:]
imaging_timestamps_0 = nwbfile.processing['ophys']['deconvolved_activity_plane_0'].timestamps

print('Shape of neuronal data for each imaging Plane')
print(deconv_0.T.shape) # Neurons | Timesteps
time_diffs_0 = np.diff(imaging_timestamps_0)
mean_time_step_0 = np.mean(time_diffs_0)
pos = bhv['frame_aligned_position']['frame_aligned_forward_and_lateral_position']

movements = np.zeros(imaging_timestamps_0.shape)

# Loop through imaging timestamps and categorize behaviors
for i, timestamp in enumerate(imaging_timestamps_0):
    # Calculate the index directly from the pattern (every 5th index)
    index = i * 5

    # Get the forward and lateral positions at the current index
    forward_pos, lateral_pos = pos.data[index]

    # Get the forward and lateral positions at the previous index
    prev_forward_pos, prev_lateral_pos = pos.data[index - 1]

    # Calculate changes in forward and lateral positions
    forward_change = forward_pos - prev_forward_pos
    lateral_change = lateral_pos - prev_lateral_pos

    movements[i] = forward_change

movements = (movements - np.nanmin(movements)) / (np.nanmax(movements) - np.nanmin(movements))
# Now, the 'behaviors' array contains the categorized behaviors for each frame in imaging timestamps.
states = ['moving forward']
movements = np.nan_to_num(movements, nan=0)
print(movements[:10])

#mouse_plane0_complete = Database(neuron_traces=deconv_0.T,
#                                 behavior=movements,
#                                 states=states,
#                                 fps=mean_time_step_0)

worm_num = 1
matlab = Loader(worm_num)
data = Database(*matlab.data)
data.neuron_traces = deconv_0.T
data.B = movements
data.states = 1
data.fps = mean_time_step_0

time, newX = preprocess_data(deconv_0, mean_time_step_0)
X_, B_ = prep_data(newX, movements, win=15)
model = BundDLeNet(latent_dim=3, behaviors=1)
model.build(input_shape=X_.shape)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
loss_array = train_model(
    X_,
    B_,
    model,
    optimizer,
    gamma=0.9,
    n_epochs=500,
    discrete=False
)

vs = Visualizer(data, model.tau, transform=False)
vs.X_ = X_
vs.B_ = B_
vs.model = model
vs.loss_array = loss_array
vs.tau_model = model.tau
vs.bn_tau = True
# I need to do this later, since X_ is not defined yet
vs._transform_points(vs.mapping)
#vs.useBundDLePredictor()

hsv_colors = np.zeros((len(movements), 3))
hsv_colors[:, 0] = 0  # Hue (red)
hsv_colors[:, 1] = 1  # Saturation (full saturation)
hsv_colors[:, 2] = 1 - movements  # Value (inverse of the values)

# Convert HSV colors to RGB
rgb_colors = plt.cm.colors.hsv_to_rgb(hsv_colors)

data.colors = rgb_colors
vs.plot_mapping(show_legend=True)
vs.make_comparison(show_legend=True)




exit()
# Do some cool plots

worm_num = 1
matlab = Loader(worm_num)
data = Database(*matlab.data)
#data.behavioral_state_diagram(save=True, show=False, adj_matrix=True)
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
lg = LogisticRegression()
data.fit_model(rf, ensemble=False)

vs1 = data.createVisualizer(PCA(n_components=3))

vs1.make_comparison(show_legend=True)
data.fit_model(et, ensemble=True)
vs1.make_comparison(show_legend=True)
data.fit_model(lg, ensemble=False)
vs1.make_comparison(show_legend=True)
exit()
vs1.plot_mapping()

data_small = vs1.use_mapping_as_input()
logreg = LogisticRegression()
data_small.fit_model(logreg)

data_small.cluster_BPT(nrep=10, max_clusters=15)