from scripts.classes import *
from sklearn.metrics import accuracy_score
import pickle
from scripts.classes import *

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

logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg, binary=True)
exit()
data.cluster_BPT(nrep=30, max_clusters=7, plot_markov=False)

vs = data.createVisualizer()
#vs.plot3D_mapping(quivers=True)
#vs.make_movie(quivers=True)
vs.behavioral_state_diagram(cog_stat_num=3, adj_matrix=False)


vs.behavioral_state_diagram(cog_stat_num=7, adj_matrix=False, offset=5)
vs.behavioral_state_diagram(cog_stat_num=4, adj_matrix=False)

#vs.behavioral_state_diagram(cog_stat_num=1)
exit()

# testing mapping
#logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg)
#data.test_markov(nrep=3, sim_markov=30, max_clusters=25)
vs = data.createVisualizer()
#vs.make_movie(dim_red=PCA(n_components=3))
#vs.make_comparison(dim_red=TSNE(n_components=3))

vs.make_comparison(dim_red=NMF(n_components=3))
vs.make_comparison(show_legend=True)

vs.attachBundleNet(epochs=750)
vs.make_comparison(show_legend=True)
exit()
print('END')
vs.make_movie(dim_red=None)
print('END')

vs.make_movie(dim_red=PCA(n_components=3))
print('END')

vs.make_movie()
vs.make_comparison()
exit()
logreg1 = LogisticRegression(multi_class='ovr', max_iter=1000)
#data.fit_model(logreg1, markov_test=True, nrep=10, sim_markov=30, max_clusters=25)
#data.plot_markov()
#print(data.yp_map)
#logreg2 = LogisticRegression(multi_class='multinomial', max_iter=1000)
#data.fit_model(logreg2, markov_test=False, nrep=10, sim_markov=15, max_clusters=40)

data.fit_28_model(logreg, markov_test=True, nrep=10, sim_markov=30, max_clusters=25)
data.plot_markov()
