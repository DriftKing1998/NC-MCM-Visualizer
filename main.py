from scripts.classes import *


db = Database(neuron_traces=[[1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                             [2, 4, 3, 1, 1, 3, 3, 4, 1, 1],
                             [1, 2, 1, 2, 2, 1, 1, 2, 1, 2],
                             [1, 1.2, 1.2, 1, 1.2, 1.2, 1.2, 1.2, 1.2, 1]],
              behavior=[0, 1, 1, 0, 2, 2, 2, 1, 1, 0],
              neuron_names=['neuron1', 'neuron2', 'neuron3', 'neuron4'],
              fps=1)

logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
db.fit_model(logreg, binary=True)
db.cluster_BPT(nrep=10, max_clusters=4, plot_markov=False)
db.step_plot(clusters=2)
exit()

### Comparable Embeddings
bundle_model.load_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
weights_T_Y = bundle_model.T_Y.get_weights()
weights_predictor = bundle_model.predictor.get_weights()
