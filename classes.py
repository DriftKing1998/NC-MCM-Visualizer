# import MyBundle
from functions import *
# from BunDLeNet import *
from MyBundle import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional, List, Union
import mat73


class Database:

    def __init__(self, data_set_no, sep=',', verbose=0):
        """
        Reads in the data from the all files corresponding to the selected dataset.
        It stores all values into numpy arrays.

        Args:
            dataset_name (string): Defines which CSV-files will be read.
            sep (string): Seperator to split the CSV-files.
            verbose (int): Defines what will be printed out.

        Returns:
            None

        """
        self.data_set_no = data_set_no
        data_dict = mat73.loadmat('data/NoStim_Data.mat')
        data = data_dict['NoStim_Data']
        deltaFOverF_bc = data['deltaFOverF_bc'][self.data_set_no]
        derivatives = data['derivs'][self.data_set_no]
        NeuronNames = data['NeuronNames'][self.data_set_no]
        fps = data['fps'][self.data_set_no]
        States = data['States'][self.data_set_no]

        self.B = np.sum([n * States[s] for n, s in enumerate(States)], axis=0).astype(
            int)  # making a single states array in which each number corresponds to a behaviour
        self.states = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        # with open(f'data/worm_{data_set_no}_neurons.csv', 'r') as neuronfile:
        #    self.neuron_names = np.asarray(neuronfile.read().strip().split(sep))
        # with open(f'data/worm_{data_set_no}_x.csv', 'r') as featurefile:
        #    all_neuron_traces = featurefile.read().strip().split('\n')
        #    self.neuron_traces = np.asarray([x.split(sep) for x in all_neuron_traces]).T.astype(float)
        # with open(f'data/worm_{data_set_no}_y.csv', 'r') as labelfile:
        #    self.B = np.asarray(labelfile.read().strip().split(sep)).astype(int)
        # with open(f'data/worm_{data_set_no}_states.csv', 'r') as statesfile:
        #    self.states = np.asarray(statesfile.read().strip().split(sep))

        if len(self.B) != self.neuron_traces.shape[1] or len(self.neuron_names) != self.neuron_traces.shape[0]:
            print('Error')
        if verbose == 1:
            print(f'The dataset \'worm_{data_set_no}\' has been loaded successfully.\nIt has: '
                  f'{self.neuron_traces.shape[0]} neurons and {self.neuron_traces.shape[1]} observations')

    def exclude_neurons(self, exclude_neurons):
        """
        Excludes specified neurons from the database.

        Args:
            exclude_neurons (list): List of neuron names to exclude.

        Returns:
            None

        """
        neuron_names = self.neuron_names
        mask = np.zeros_like(self.neuron_names, dtype='bool')
        for exclude_neuron in exclude_neurons:
            mask = np.logical_or(mask, neuron_names == exclude_neuron)
        mask = ~mask
        amount = len(mask) - np.count_nonzero(mask)
        self.neuron_traces = self.neuron_traces[mask]
        self.neuron_names = self.neuron_names[mask]

        print(f'{amount} neurons have been removed.')

    def createVisualizer(self):
        vs = Visualizer(self.neuron_traces, self.B, xlabs=self.neuron_names, blabs=self.states, fps=self.fps)
        return vs

    def loadBundleVisualizer(self, l_dim=3):
        vs = Visualizer(self.neuron_traces, self.B, xlabs=self.neuron_names, blabs=self.states, fps=self.fps)
        time, newX = preprocess_data(vs.X.T, vs.fps)
        vs.X_, vs.B_ = prep_data(newX, vs.B, win=15)
        vs.model = BunDLeNet(latent_dim=l_dim)
        vs.model.build(input_shape=vs.X_.shape)
        vs.model.load_weights('data/generated/BunDLeNet_model_worm_' + str(self.data_set_no))
        vs.tau_model = vs.model.tau
        return vs


class Visualizer():

    def __init__(self,
                 X: Union[List[List[float]], np.ndarray],
                 B: Union[List[int], List[str], np.ndarray],
                 xlabs: Optional[Union[List[str], np.ndarray]] = None,
                 blabs: Optional[Union[List[str], np.ndarray]] = None,
                 fps: float = None):
        """
        Takes values for features(neurons), labels(behaviors) and their corresponding names. If B is a list of strings,
        those are taken as blabs and blabs is ignored.

        Args:
            X (ndarray): 2D array with each row corresponding to a neuron and each column is a timeframe
            B (ndarray): 1D array with behavior encoded as integer.
            xlabs (ndarray): 1D array with the names of the neurons.
            blabs (ndarray): 1D array with translation for behavior.
            fps (float): gives the recording fps.

        Returns:
            None

        """
        # Input Preprocessing
        X = np.asarray(X)
        B = np.asarray(B)

        # If B is not an integer array we have to transform it into one
        if not (np.issubdtype(B.dtype, int) or np.issubdtype(B.dtype, np.integer)):
            print('B has been transformed and a blabs has been created')
            newB, blabs = make_integer_list(B)
            B = np.asarray(newB)
            blabs = np.asarray(blabs).astype(str)

        # If no labs are given, they are generated
        if xlabs is None:
            xlabs = np.asarray(range(X.shape[0])).astype(str)
        else:
            xlabs = np.asarray(xlabs)
        if blabs is None:
            blabs = np.asarray(np.unique(B)).astype(str)
        else:
            blabs = np.asarray(blabs)

        # Setting Attributes
        self.X = X
        self.B = B
        self.xlabs = xlabs
        self.blabs = blabs
        self.tau_model = None
        self.model = None
        self.fps = fps

        # generate a color-dictionary for all states and generate the colors
        self.colordict = dict(zip(np.unique(self.B), generate_equidistant_colors(len(self.blabs))))
        self.colors = [self.colordict[val] for val in self.B]

    def plotting_neuronal_behavioural(self, vmin=0, vmax=2):
        fig, axs = plt.subplots(2, 1, figsize=(10, 4))
        self._neurons(ax=axs[0])
        self._behavior(ax=axs[1])
        plt.show()

    def _behavior(self, ax=None, sample=None):

        ### Sample selection ha to be made ###

        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        cmap = plt.get_cmap('Pastel1', np.max(self.B) - np.min(self.B) + 1)
        im1 = ax.imshow([self.B], cmap=cmap, vmin=np.min(self.B) - 0.5, vmax=np.max(self.B) + 0.5, aspect='auto')
        # tell the colorbar to tick at integers
        cax = ax.get_figure().colorbar(im1, ticks=np.arange(np.min(self.B), np.max(self.B) + 1))
        if len(np.unique(self.B)) == len(self.blabs):
            cax.ax.set_yticklabels(self.blabs)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Behaviour")
        ax.set_yticks([])
        if sample:
            ax.set_title(f'Sample no#{sample}')

        if show:
            plt.show()

    def _neurons(self, ax=None, sample=None, vmin=0, vmax=2):

        ### Sample selection ha to be made ###

        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        im0 = ax.imshow(self.X, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None')
        # tell the colorbar to tick at integers
        # plt.colorbar(im0)
        ax.get_figure().colorbar(im0)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Neuronal activation")

        if show:
            plt.show()

    def _transform_points(self):
        if self.tau_model is None:
            print('CREATE PCA MODEL')
            pca = PCA(n_components=3)
            pca.fit(self.X.T)
            self.tau_model = pca
            transformed_points = pca.transform(self.X.T)
            print(transformed_points.shape)
        else:
            if isinstance(self.model, BunDLeNet):
                print('HAVE BUNDLE MODEL')
                print(self.tau_model.input_shape, ' is input shape, we have: ', self.X_[:, 0].shape)

                transformed_points = self.tau_model.predict(self.X_[:, 0])
                print(transformed_points.shape)

            else:
                print('HAVE PCA MODEL')
                print(self.X.T.shape, ' is input shape, we have: ', self.X.T.shape)

                transformed_points = self.tau_model.transform(self.X.T)
                print(transformed_points.shape)

        self.x, self.y, self.z = transformed_points.T

    def plot3D_mapping(self, use_tau=True, show_legend=False):
        if use_tau and self.tau_model is not None:
            self.tau_model = self.model.tau
        else:
            self.tau_model = None

        self._transform_points()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # We need to trim labs and colors if we have a Bundle
        if len(self.x) < len(self.colors):
            window = len(self.colors) - len(self.x)
            colors = self.colors[window:]
        else:
            colors = self.colors

        ax.scatter(self.x, self.y, self.z, label=self.blabs, s=2, alpha=0.5, color=colors)
        ax.set_xlabel('Axes 1')
        ax.set_ylabel('Axes 2')
        ax.set_zlabel('Axes 3')

        # plot the legend if wanted
        if show_legend:
            legend_elements, y_labels = self._generate_legend(self.blabs)
            ax.legend(legend_elements, y_labels, loc='upper left', fontsize='x-small')

        ax.grid(False)
        plt.show()

    ################## A
    def _generate_legend(self, classifier, diff=False):
        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colordict[idx],
                                      markersize=10) for idx, lab in enumerate(self.blabs)]

        # if the legend for the difference plot is requested
        if diff:
            y_labels_diff = {
                label: {wrong: count for wrong, count in self.diff_label_counts[c_idx].items() if count}
                for c_idx, label in enumerate(self.blabs)
            }

            y_labels_list = [
                (r'$\mathbf{' + c_behav + '}$' +
                 (' to ' + '; '.join([r'$\mathbf{' + w_behav + '}$' + f'({amount})' for w_behav, amount in
                                      val.items()]) if val else ' always correct'))
                for c_behav, val in y_labels_diff.items()
            ]
            return legend_elements, y_labels_list

        # if a normal legend is requested
        y_labels = [r'$\mathbf{' + behavior + '}$' + f' ({list(self.B).count(idx)})' for
                    idx, behavior in enumerate(self.blabs)]

        return legend_elements, y_labels

    ### HELP
    def _generate_label_counts(self):
        merged_list = [False if self.B[i] == self.Y_pred[i] else self.B[i] for i in range(len(self.B))]
        self.diff_colors_pred = [self.colordict[val] if val else "lightgrey" for val in merged_list]
        self.colors_pred = [self.colordict[val] for val in self.Y_pred]

        # Initialize a dictionary to store and count correct and false predictions for each label
        self.pred_label_counts = {}
        self.true_label_counts = {}
        self.diff_label_counts = {}
        for idx, wrong_predict in enumerate(merged_list):
            pred_label = self.Y_pred[idx]
            true_label = self.B[idx]

            if pred_label not in self.pred_label_counts:
                self.pred_label_counts[pred_label] = {True: 0, False: 0}
            if true_label not in self.true_label_counts:
                self.true_label_counts[true_label] = {True: 0, False: 0}
            if true_label not in self.diff_label_counts:
                self.diff_label_counts[true_label] = {lab: 0 for lab in self.blabs}

            if wrong_predict:
                self.pred_label_counts[pred_label][False] += 1
                self.true_label_counts[true_label][False] += 1
                self.diff_label_counts[true_label][self.blabs[pred_label]] += 1
            else:
                self.pred_label_counts[pred_label][True] += 1
                self.true_label_counts[true_label][True] += 1

    def attachBundleNet(self, l_dim=3, train=True, epochs=2000):
        time, newX = preprocess_data(self.X.T, self.fps)
        self.X_, self.B_ = prep_data(newX, self.B, win=15)

        self.model = BunDLeNet(latent_dim=l_dim)
        self.model.build(input_shape=self.X_.shape)

        if train:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
            self.loss_array = train_model(
                self.X_,
                self.B_,
                self.model,
                optimizer,
                gamma=0.9,
                n_epochs=epochs
            )

            self.tau_model = self.model.tau

        return self.model

    def plot_loss(self):
        if self.model is not None:
            plt.figure()
            for i, label in enumerate(
                    ["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$", "Total loss $\mathcal{L}$"]):
                plt.semilogy(self.loss_array[:, i], label=label)
            plt.legend()
            plt.show()
        else:
            print('No model was trained.')

    def train_model(self, epochs=2000, learning_rate=0.001):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.loss_array = train_model(
            self.X_,
            self.B_,
            self.model,
            optimizer,
            gamma=0.9,
            n_epochs=epochs
        )
        self.tau_model = self.model.tau
