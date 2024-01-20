from scripts.classes import *
import numpy as np

db = Database(neuron_traces=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                      behavior=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      neuron_names=['neuron1', 'neuron2', 'neuron3', 'neuron4'],
                      fps=1)

# Without mapping
visualizer_no_mapping = db.createVisualizer(window=3, epochs=20)
print(visualizer_no_mapping.mapping)
#self.assertIsNotNone(visualizer_no_mapping)
#self.assertIsNone(visualizer_no_mapping.mapping)
#self.assertIsNone(visualizer_no_mapping.tau_model)
exit()
"""
P = np.asarray([[0.65, 0.1, 0.2, 0.05, 0],
                [0.4, 0.1, 0.05, 0.4, 0.05],
                [0.15, 0.75, 0.10, 0.05, 0],
                [0.5, 0.1, 0.15, 0.05, 0.2],
                [0.025, 0.025, 0.025, 0.025, 0.9]])

Y, _ = simulate_markovian(P=P, M=1000)
print(Y)

X = np.random.rand(150, len(Y))
blabs = ['A', 'B', 'C', 'D', 'E']

influence_factor1 = 0.025
influence_factor2 = 0.005
influence_factor3 = 0.01
influence_factor4 = 0.02
# Use the values in Y to introduce structure
for i, y_value in enumerate(Y):
    X[:50, i] += influence_factor1 * (y_value % 5)  # Adjust the expression as needed
    X[50:100, i] += influence_factor2 * (y_value % 5)  # Adjust the expression as needed
    X[100:125, i] += influence_factor3 * (y_value % 5) + pow(X[35:60, i], 2) # Adjust the expression as needed
    X[125:, i] += influence_factor4 * (y_value % 5) - pow(X[85:110, i], 2)  # Adjust the expression as needed"""

l = Loader(1)
a, b, c, d, e = l.data
data = Database(a, b, c, d, fps=10)
#data = Database(X, Y, states=blabs, fps=10)
data.plotting_neuronal_behavioural()
#data.step_plot(clusters=3, nrep=10)

logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg, binary=True)

vs = data.createVisualizer(epochs=1000)
vs.make_movie(save=True, quivers=True)