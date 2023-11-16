from scripts.classes import *
from IPython.display import display

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
exit()
data.exclude_neurons(b_neurons)
vs = data.loadBundleVisualizer()
#vs = data.createVisualizer()

#svm_model = SVC()
#logistic_model = LogisticRegression()

#dim_red = PCA(n_components=2)
#dim_red = TSNE(n_components=3)
#dim_red = SparseRandomProjection(n_components=3)
#dim_red = NMF(n_components=3)

vs.make_movie(grid_off=True, interval=20, show_legend=True, quivers=False, save=False)
vs.save_gif('second_movie')

#vs.plotting_neuronal_behavioural()
#vs.plot3D_mapping(show_legend=True, grid_off=True, quivers=False)
