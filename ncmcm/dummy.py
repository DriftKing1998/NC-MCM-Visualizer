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
data = Database(*matlab.data)
#seq = simulate_random_sequence(10000, 10)
lg = LogisticRegression(max_iter=1000)
data.fit_model(lg, ensemble=True)
vs = data.createVisualizer()
vs.make_movie(draw=False)
# data.fit_model(lg, ensemble=True, cv_folds=10)
# data.fit_model(lg, ensemble=True, cv_folds=15)
# data.fit_model(lg, ensemble=False, cv_folds=5)
# data.fit_model(lg, ensemble=False, cv_folds=10)
# data.fit_model(lg, ensemble=False, cv_folds=15)
#data.cluster_BPT(nrep=10, max_clusters=40, stationary=True)
data.cluster_BPT_single(nrep=10, nclusters=5, stationary=True)
data.behavioral_state_diagram(cog_stat_num=5, interactive=True)
data.cluster_BPT_single(nrep=10, nclusters=5, stationary=True)
data.behavioral_state_diagram(cog_stat_num=5, interactive=True)

#data.cluster_BPT_single(nrep=30, nclusters=3, stationary=True)
data._plot_markov(stationary=True)
data.behavioral_state_diagram(cog_stat_num=3, interactive=True)
data.behavioral_state_diagram(cog_stat_num=5, interactive=True)
exit()
data.cluster_BPT_single(nrep=30, nclusters=4, stationary=False)
data.cluster_BPT_single(nrep=30, nclusters=2, stationary=False)
data.behavioral_state_diagram(cog_stat_num=2, interactive=True)
data.behavioral_state_diagram(cog_stat_num=4, interactive=True)
exit()
pca = PCA(n_components=3)
vs = data.createVisualizer(pca)

res = []
for i in range(30):
    start = time.time()
    vs.plot_mapping(show_legend=True, show=False)
    end = time.time()
    print(end-start)
    res.append(end-start)
print(f'Mean time is {np.mean(res)} seconds.')
#vs = data.createVisualizer()



exit()
vs.make_movie()
data.plotting_neuronal_behavioral()
data.cluster_BPT_single(nclusters=2, nrep=3, chunks=None)
#data.cluster_BPT_single(nclusters=3, nrep=30, chunks=None)
#data.step_plot(clusters=5, nrep=30)
data.behavioral_state_diagram(cog_stat_num=2, interactive=True, adj_matrix=True)
#data.behavioral_state_diagram(cog_stat_num=2, interactive=True)
#data.behavioral_state_diagram(cog_stat_num=3, interactive=True)
#data.behavioral_state_diagram(cog_stat_num=5, interactive=True)


data.fit_model(lg, ensemble=False)
data.cluster_BPT(max_clusters=5, nrep=7, stationary=True, chunks=None)
data.step_plot(clusters=4)
print(data.p_memoryless.shape)
data.cluster_BPT_single(5, nrep=5, stationary=True, chunks=None)
data._plot_markov(stationary=True)
print(data.fps)
print(data.neuron_traces.shape)

exit()

time, newX = preprocess_data(data.neuron_traces.T, data.fps)
sum_array = np.sum(newX, axis=1)
noise = np.random.normal(0, 0.1, size=sum_array.shape)
sum_array_with_noise = sum_array + noise
B = (sum_array_with_noise - np.min(sum_array_with_noise)) / (np.max(sum_array_with_noise) - np.min(sum_array_with_noise))

X_, B_ = prep_data(newX, B, win=15)
model = BundDLeNet(latent_dim=3, behaviors=1)
model.build(input_shape=X_.shape)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
loss_array = train_model(
    X_,
    B_,
    model,
    optimizer,
    gamma=0.9,
    n_epochs=1000
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
vs.useBundDLePredictor()

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