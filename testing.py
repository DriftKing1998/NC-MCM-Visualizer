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

data.step_plot(clusters=5, nrep=10)
vs = data.createVisualizer()
vs.behavioral_state_diagram(cog_stat_num=3)
vs.behavioral_state_diagram(cog_stat_num=1)
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

# end

arr = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
com = list(combinations(arr, 2))
Y = data.B
X = data.neuron_traces.T
models = []

for class_mapping in com:
	# make a mask using the current combination
	mapped_Y = np.array([y if y in class_mapping else -1 for y in Y])
	mask = mapped_Y != -1

	X_train_filtered = X[mask]
	mapped_Y_filtered = mapped_Y[mask]

	# Now, you can train your logistic regression model for 'A' vs. 'B'
	logreg = LogisticRegression(solver='liblinear', max_iter=1000)
	logreg.fit(X_train_filtered, mapped_Y_filtered)
	models.append(logreg)

results = np.zeros((28, X.shape[0]))

for idx, model in enumerate(models):
	pred = model.predict_proba(X)[:,0]
	print(pred[:5])
	results[idx, :] = pred

results = results.astype(int)
majority_vote = [np.bincount(results[:, col]).argmax() for col in range(results.shape[1])]

print(len(majority_vote))
print(majority_vote)

print(f'ACCURACY: {accuracy_score(majority_vote, Y)}')