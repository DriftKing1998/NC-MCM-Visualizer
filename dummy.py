from scripts.classes import *
from IPython.display import display
import os
import pickle
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE, SpectralEmbedding

#os.chdir('..')
print(os.getcwd())

WORMS = []
m_data = []
for i in range(5):
	with open(f'data/pickles/data_worm_{i}.pkl', 'rb') as file:
		worm = pickle.load(file)
		WORMS.append(worm)
		m_data.append(np.mean(worm.p_memoryless, axis=1))

ml = Loader(0)
data = Database(*ml.data)
vs = data.createVisualizer(epochs=2500)

vs.make_movie(save=True, draw=False, quivers=True, interval=100, show_legend=True)
exit()
average_markov_plot(np.asarray(m_data))

for w in WORMS:
	w._plot_markov()
average_markov_plot(np.asarray(m_data))

exit()
# linear manifold embeddings
# (i) PCA
pca = PCA(n_components=3)
# (ii) MDS
mds = MDS(n_components=3, normalized_stress='auto')
# non-linear manifold embeddings
# (i) Isomap
isomap = Isomap(n_components=3, n_neighbors=50)
# (ii) LLE
lle = LocallyLinearEmbedding(n_components=3, n_neighbors=50)
# (iii) LEM
lem = SpectralEmbedding(n_components=3)
# (iv) t-SNE
tsne = TSNE(n_components=3)

dim_reds = [pca, mds, isomap, lle, lem, tsne]

for worm in WORMS:
	vs = worm.createVisualizer(dim_reds[0])
	vs.plot_mapping(show_legend=True)

	for dim_red in dim_reds[1:]:
		vs.change_mapping(dim_red)
		vs.plot_mapping(show_legend=True)
