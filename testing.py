from collections import Counter
from classes import *
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection


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
#vs = data.loadBundleVisualizer()
vs = data.createVisualizer()

#svm_model = SVC()
#logistic_model = LogisticRegression()

#dim_red = PCA(n_components=2)
#dim_red = TSNE(n_components=3)
#dim_red = SparseRandomProjection(n_components=3)
#dim_red = NMF(n_components=3)

vs.plotting_neuronal_behavioural()
vs.plot3D_mapping(show_legend=True, grid_off=True, quivers=False)
