# import MyBundle
from scripts.functions import *
# from BunDLeNet import *
from scripts.MyBundle import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from typing import Optional, List, Union
import mat73
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.animation as anim  # FuncAnimation
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.manifold import TSNE

import networkx as nx
from itertools import combinations
from pyvis.network import Network


class Database:

    def __init__(self, data_set_no, sep=',', verbose=0):
        """
        Reads in the data from the all files corresponding to the selected dataset.
        It stores all values into numpy arrays.

        :param data_set_no: Defines which CSV files will be read.
        :type data_set_no: int

        :param sep: Separator to split the CSV files.
        :type sep: string

        :param verbose: Defines the verbosity level (0 for minimal output).
        :type verbose: int
        """

        self.data_set_no = data_set_no
        data_dict = mat73.loadmat('/Users/michaelhofer/Documents/GitHub/NeuronVisualizer2.0/data/NoStim_Data.mat')
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
        self.pred_model = None
        self.B_pred = None
        self.yp_map = None
        self.p_markov = None
        self.xc = None

        self.colordict = dict(zip(np.unique(self.B), generate_equidistant_colors(len(self.states))))
        self.colors = [self.colordict[val] for val in self.B]

        if len(self.B) != self.neuron_traces.shape[1] or len(self.neuron_names) != self.neuron_traces.shape[0]:
            print('Error')
        if verbose == 1:
            print(f'The dataset \'worm_{data_set_no}\' has been loaded successfully.\nIt has: '
                  f'{self.neuron_traces.shape[0]} neurons and {self.neuron_traces.shape[1]} observations')

    def exclude_neurons(self, exclude_neurons):
        """
        Excludes specified neurons from the database.

        :param exclude_neurons: List of neuron names to exclude.
        :type exclude_neurons: list
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
        vs = Visualizer(Data=self)
        return vs

    def loadBundleVisualizer(self, weights_path=None, l_dim=3):
        vs = Visualizer(Data=self)
        time, newX = preprocess_data(vs.data.neuron_traces.T, vs.data.fps)
        vs.X_, vs.B_ = prep_data(newX, vs.data.B, win=15)
        vs.model = BunDLeNet(latent_dim=l_dim)
        vs.model.build(input_shape=vs.X_.shape)
        if weights_path is None:
            vs.model.load_weights('data/generated/BunDLeNet_model_worm_' + str(self.data_set_no))
        else:
            vs.model.load_weights(weights_path)
        vs.tau_model = vs.model.tau
        vs.bundle_tau = True
        return vs

    def fit_model(self, base_model, prob_map=True, binary=False):
        if not hasattr(base_model, 'fit'):
            print('Model has no method \'fit\'.')
            return None

        # For a binary regression by hand
        if binary:
            self.pred_model = CustomEnsembleModel(base_model).fit(self.neuron_traces.T, self.B)
            self.B_pred = self.pred_model.predict(self.neuron_traces.T)
            print("Accuracy:", accuracy_score(self.B, self.B_pred))
            if prob_map:
                # It should not make a difference if we use 28 or 56 dimensions
                self.yp_map = self.pred_model.predict_proba(self.neuron_traces.T)
                print(f'Probability map has shape: {self.yp_map.shape}')

        # For a multiclass regression
        else:
            self.pred_model = base_model.fit(self.neuron_traces.T, self.B)
            self.B_pred = self.pred_model.predict(self.neuron_traces.T)
            print("Accuracy:", accuracy_score(self.B, self.B_pred))
            if prob_map:
                # get probabilities and weights
                self.yp_map = self.pred_model.predict_proba(self.neuron_traces.T)
                print(f'Probability map has shape: {self.yp_map.shape}')

        return True

    def _multiclasstrain(self, b_model, class_mapping):
        # make a mask using the current combination ('A' vs 'B')
        mapped_B = np.array([b if b in class_mapping else -1 for b in self.B])
        mask = mapped_B != -1
        # apply mask to the dataset and only use instances of 'A' or 'B' to train
        X_train_filtered = self.neuron_traces.T[mask]
        mapped_B_filtered = mapped_B[mask]
        # We train the logistic regression model to differentiate 'A' vs. 'B'
        b_model.fit(X_train_filtered, mapped_B_filtered)
        return b_model

    def cluster_BPT(self, nrep=200, max_clusters=20, sim_markov=200, plot_markov=True):
        if self.yp_map is None:
            print(f'You first need to fit a model (eg. Logistic Regression), '
                  f'which will be used to map to behavioral probability trajectories.\n'
                  f'Use \'.fit_model(<your_model>)\' on this instance before.')
            return False

        self.p_markov = np.zeros((max_clusters, nrep))
        M = self.yp_map.shape[0]
        self.xc = np.zeros((M, max_clusters, nrep))

        for reps in range(nrep):
            print("Testing markovianity - repetition ", reps + 1)
            for nrclusters in range(max_clusters):
                # k-means
                clusters = KMeans(n_clusters=nrclusters + 1, n_init="auto").fit(self.yp_map)
                xctmp = clusters.labels_
                p, _ = markovian(xctmp, K=sim_markov)
                self.p_markov[nrclusters, reps] = p
                self.xc[:, nrclusters, reps] = xctmp

        if plot_markov:
            self._plot_markov()

        return True

    def _plot_markov(self):
        fig, ax = plt.subplots()
        data = self.p_markov[:, :].T

        # Plotting
        ax.boxplot(data)
        ax.set_title(f'Probability of being a Markov process for worm {self.data_set_no + 1}')
        ax.set_xlabel('Number of States/Clusters')
        ax.set_ylabel('Probability')
        ax.axhline(0.05)
        plt.tight_layout()
        plt.show()

    def step_plot(self, clusters=5, nrep=10, sim_markov=200, save=True, show=True):
        if self.p_markov is None:
            self.fit_model(LogisticRegression(solver='lbfgs', max_iter=1000), binary=True)
            self.cluster_BPT(nrep=nrep, max_clusters=clusters, sim_markov=sim_markov, plot_markov=False)

        # Neuronal trajectories preprocessing
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        pca = PCA(n_components=2)
        plot_values = pca.fit_transform(self.neuron_traces.T)
        x_nt, y_nt = plot_values.T
        handles = []
        for idx, state in enumerate(self.states):
            patch = mpatches.Patch(color=self.colordict[idx], label=state)
            handles.append(patch)

        # Behavioral probability trajectories preprocessing
        pca = PCA(n_components=2)
        plot_values = pca.fit_transform(self.yp_map)
        x_bpt, y_bpt = plot_values.T
        colordict_cog = dict(zip(list(range(clusters)), generate_equidistant_colors(clusters)))
        best_clustering_idx = np.argmax(self.p_markov[clusters - 1, :])  # according to mr.markov himself
        best_clustering = self.xc[:, clusters - 1, best_clustering_idx].astype(int)
        cog_colors = [colordict_cog[l] for l in best_clustering]
        handles_cog = []
        for idx in range(clusters):
            patch = mpatches.Patch(color=colordict_cog[idx], label=f'C{idx + 1}')
            handles_cog.append(patch)

        # UPPER LEFT PLOT
        ax[0, 0] = self._add_quivers2D(ax[0, 0], x_nt, y_nt, None)
        ax[0, 0].legend(handles=handles, loc='best')
        ax[0, 0].set_title('Neuronal trajectories with behavioral labels')

        # UPPER RIGHT PLOT
        ax[0, 1] = self._add_quivers2D(ax[0, 1], x_bpt, y_bpt, None)
        ax[0, 1].set_title('Behavioral probability trajectories with behavioral labels')

        # LOWER LEFT PLOT
        ax[1, 0] = self._add_quivers2D(ax[1, 0], x_bpt, y_bpt, colors=cog_colors)
        ax[1, 0].set_title('Behavioral probability trajectories with cognitive labels')

        # LOWER RIGHT PLOT
        ax[1, 1] = self._add_quivers2D(ax[1, 1], x_nt, y_nt, colors=cog_colors)
        ax[1, 1].legend(handles=handles_cog, loc='best')
        ax[1, 1].set_title('Neuronal trajectories with cognitive labels')

        # GENERAL
        fig.suptitle(f'Worm #{self.data_set_no + 1} with {clusters} cognitive states')
        if save:
            plt.savefig(f'data/plots/step_plot_worm_{self.data_set_no + 1}.png', format='png')
        if show:
            plt.show()

    def _add_quivers2D(self, ax, x, y, colors=None):
        if colors is None:
            colors = self.colors[:-1]
        dx = np.diff(x)  # Differences between x coordinates
        dy = np.diff(y)  # Differences between y coordinates
        ax.quiver(x[:-1], y[:-1], dx, dy, color=colors, alpha=0.5)
        return ax


class Visualizer():

    def __init__(self,
                 # X: Union[List[List[float]], np.ndarray],
                 # B: Union[List[int], List[str], np.ndarray],
                 Data: Database):
        # B_pred: Union[List[int], List[str], np.ndarray],
        # xlabs: Optional[Union[List[str], np.ndarray]] = None,
        # blabs: Optional[Union[List[str], np.ndarray]] = None,
        # fps: float = None):
        """
        Takes values for features (neurons), labels (behaviors), and their corresponding names. If B is a list of strings,
        those are taken as blabs, and blabs is ignored.

        :param X: ndarray
        :type X: 2D array with each row corresponding to a neuron and each column is a timeframe.

        :param B: ndarray
        :type B: 1D array with behavior encoded as integers.

        :param xlabs: ndarray
        :type xlabs: 1D array with the names of the neurons.

        :param blabs: ndarray
        :type blabs: 1D array with translation for behavior.

        :param fps: float
        :type fps: gives the recording fps.

        Returns:
            None
        """

        # Setting Attributes
        self.data = Data

        # If B is not an integer array we have to transform it into one
        if not (np.issubdtype(self.data.B.dtype, int) or np.issubdtype(self.data.B.dtype, np.integer)):
            print('B has been transformed and a blabs has been created')
            newB, blabs = make_integer_list(self.data.B)
            self.data.B = np.asarray(newB)
            self.data.states = np.asarray(blabs).astype(str)

        # If no Neuron Names / State Names are given, they are generated
        if self.data.neuron_names is None:
            self.data.neuron_names = np.asarray(range(self.data.neuron_traces.shape[1])).astype(str)
        else:
            self.data.neuron_names = np.asarray(self.data.neuron_names)
        if self.data.states is None:
            self.data.states = np.asarray(np.unique(self.data.B)).astype(str)
        else:
            self.data.states = np.asarray(self.data.states)

        # generate a color-dictionary for all states and generate the colors
        self.colordict = dict(zip(np.unique(self.data.B), generate_equidistant_colors(len(self.data.states))))
        self.colors = [self.colordict[val] for val in self.data.B]
        self.trimmed_colors = None
        # BundleNet
        self.tau_model = None
        self.bundle_tau = False
        self.model = None
        # Animation
        self.animation = None
        self.interval = None
        self.scatter = None

    ### DIAGNOSTICS ###
    def plotting_neuronal_behavioural(self, vmin=0, vmax=2):
        fig, axs = plt.subplots(2, 1, figsize=(10, 4))
        self._neurons(ax=axs[0])
        self._behavior(ax=axs[1])
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def _behavior(self, ax=None, sample=None):

        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        cmap = plt.get_cmap('Pastel1', np.max(self.data.B) - np.min(self.data.B) + 1)
        im1 = ax.imshow([self.data.B], cmap=cmap, vmin=np.min(self.data.B) - 0.5, vmax=np.max(self.data.B) + 0.5,
                        aspect='auto')
        # tell the colorbar to tick at integers
        cax = ax.get_figure().colorbar(im1, ticks=np.arange(np.min(self.data.B), np.max(self.data.B) + 1))
        if len(np.unique(self.data.B)) == len(self.data.states):
            cax.ax.set_yticklabels(self.data.states)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Behaviour")
        ax.set_yticks([])
        ax.set_title(f'Sample no#{self.data.data_set_no + 1}')

        if show:
            plt.show()

    def _neurons(self, ax=None, vmin=0, vmax=2):
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        im0 = ax.imshow(self.data.neuron_traces, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None')
        # tell the colorbar to tick at integers
        # plt.colorbar(im0)
        ax.get_figure().colorbar(im0)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Neurons")
        ax.set_title("Neuronal activation")
        if show:
            plt.show()

    def plot3D_mapping(self, dim_red=None, show_legend=False, grid_off=True, quivers=False):

        if dim_red is None:
            if self.bundle_tau:
                dim_red = self.tau_model
                print('We use the tau model from the BunDLeNet to project into 3D space.')
            else:
                dim_red = PCA(n_components=3)
                print('No mapping present. CREATING PCA MODEL ...')

        if not self._transform_points(dim_red):
            return False

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # We need to trim labs and colors if we have a Bundle
        if len(self.x) < len(self.colors):
            window = len(self.colors) - len(self.x)
            colors = self.colors[window:]
        else:
            colors = self.colors

        if quivers:
            ax = self._add_quivers3D(ax, self.x, self.y, self.z, colors=colors)
        else:
            ax.scatter(self.x, self.y, self.z, label=self.data.states, s=1, alpha=0.5, color=colors)

        ax.set_xlabel('Axes 1')
        ax.set_ylabel('Axes 2')
        ax.set_zlabel('Axes 3')

        # plot the legend if wanted
        if show_legend:
            print('HEHEEH')
            legend_elements = self._generate_legend(self.data.B)
            ax.legend(handles=legend_elements, loc='best')

        if grid_off:
            ax.grid(False)
            ax.set_axis_off()

        plt.show()

    def _transform_points(self, dim_red):
        if dim_red is None:
            print('No mapping present. CREATING PCA MODEL ...')
            dim_red = PCA(n_components=3)
            transformed_points = dim_red.fit_transform(self.data.neuron_traces.T)
        # If we are using the TAU model to map into 3D space
        elif isinstance(dim_red, tf.keras.Sequential):
            if dim_red.input_shape[1] == self.X_.shape[2]:
                transformed_points = np.asarray(dim_red(self.X_[:, 0]))
            else:
                transformed_points = np.asarray(dim_red(self.data.neuron_traces.T))
        # This happens if we give some mapping which is not a NN
        elif hasattr(dim_red, 'fit_transform'):
            if dim_red.get_params()['n_components'] == 3:
                print('HAVE mapping MODEL')
                if isinstance(dim_red, NMF):
                    scaler = MinMaxScaler(feature_range=(0, np.max(self.data.neuron_traces.T)))
                    X_scaled = scaler.fit_transform(self.data.neuron_traces.T)
                    transformed_points = dim_red.fit_transform(X_scaled)
                else:
                    transformed_points = dim_red.fit_transform(self.data.neuron_traces.T)
            else:
                print('The selected model does not project to a 3 dimensional space.')
                return False
        else:
            print('The selected mapping has no attribute \'fit_transform\'. (SKLEARN models are recommended)')
            return False

        print('Points have coordinate shape: ', transformed_points.shape)
        self.x, self.y, self.z = transformed_points.T
        return True

    def _generate_legend(self, blabs, diff=False):
        # if the legend for the difference plot is requested
        if diff:
            y_labels_diff = {
                label: {wrong: count for wrong, count in self.diff_label_counts[c_idx].items() if count}
                for c_idx, label in enumerate(self.data.states)
            }

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colordict[idx],
                                          markersize=10,
                                          label=r'$\mathbf{' + keyval[0] + '}$' + ' to ' +
                                                "; ".join(
                                                    [r"$\mathbf{" + w_behav + "}$" + f"({amount})"
                                                     for w_behav, amount in keyval[1].items()]
                                                )
                                          if keyval[1] else r'$\mathbf{' + keyval[0] + '}$' + " predictions were always correct")
                               for idx, keyval in enumerate(y_labels_diff.items())]

            return legend_elements

        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colordict[idx],
                                      markersize=10,
                                      label=r'$\mathbf{' + lab + '}$' + f' ({list(blabs).count(idx)})')
                           for idx, lab in enumerate(self.data.states)]

        return legend_elements

    def _generate_label_counts(self):
        # We need to trim labs and colors if we have a Bundle
        if len(self.x) < len(self.colors):
            window = len(self.colors) - len(self.x)
            self.trimmed_colors = self.colors[window:]

            merged_list = [False if self.data.B[i] == self.data.B_pred[i] else self.data.B[i] for i in
                           range(len(self.data.B))]
            self.diff_colors_pred = [self.colordict[val] if val else (0.8, 0.8, 0.8) for val in merged_list[window:]]
            self.colors_pred = [self.colordict[val] for val in self.data.B_pred[window:]]
            print(
                f'Length of color arrays: {len(self.trimmed_colors), len(self.colors_pred), len(self.diff_colors_pred)}')
        else:
            self.trimmed_colors = self.colors

            merged_list = [False if self.data.B[i] == self.data.B_pred[i] else self.data.B[i] for i in
                           range(len(self.data.B))]
            self.diff_colors_pred = [self.colordict[val] if val else (0.8, 0.8, 0.8) for val in merged_list]
            self.colors_pred = [self.colordict[val] for val in self.data.B_pred]

        # Initialize a dictionary to store and count correct and false predictions for each label
        self.diff_label_counts = {}
        for idx, wrong_predict in enumerate(merged_list):
            pred_label = self.data.B_pred[idx]
            true_label = self.data.B[idx]

            if true_label not in self.diff_label_counts:
                self.diff_label_counts[true_label] = {lab: 0 for lab in self.data.states}
            if wrong_predict:
                self.diff_label_counts[true_label][self.data.states[pred_label]] += 1


    def attachBundleNet(self, l_dim=3, train=True, epochs=2000):
        if self.data.fps is None:
            print('In order to attach the BundleNet \'self.data.fps\' has to have a value!')
            return False

        time, newX = preprocess_data(self.data.neuron_traces.T, self.data.fps)
        self.X_, self.B_ = prep_data(newX, self.data.B, win=15)

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
            self.bundle_tau = True

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
        self.bundle_tau = True

    def _add_quivers3D(self, ax, x, y, z, colors=None):
        if colors is None:
            colors = self.colors[:-1]

        dx = np.diff(x)  # Differences between x coordinates
        dy = np.diff(y)  # Differences between y coordinates
        dz = np.diff(z)  # Differences between z coordinates
        # we do this so each arrowhead has the same size independent of the size of the arrow
        lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        lengths = 0.05 / lengths
        for idx in range(len(dx)):
            ax.quiver(x[idx], y[idx], z[idx], dx[idx], dy[idx], dz[idx], color=colors[idx],
                      arrow_length_ratio=lengths[idx], alpha=0.8, linewidths=0.8)

        return ax

    def make_movie(self, interval=None, dim_red=None, save=False, show_legend=False, grid_off=True, quivers=False):
        """
        Makes a movie out of each frame in the imaging data. It uses the tau model or a model given as a parameter to
        map the data to a 3-dimensional space.

        :param interval :  Interval between frames in milliseconds.
        :type interval : int

        :param dim_red : Specifies the latent dimension mapping used for the 3D plot.
        :type dim_red : tf.keras.Sequential() with 3-dimensional output or any scikit learn dimensionality reduction

        :param save : If the movie should be saved.
        :type save : boolean

        :param show_legend : If the legend is visible.
        :type show_legend : boolean

        :param grid_off : If the grid and axes are visible.
        :type grid_off : boolean

        :param quivers : If quivers are used instead of scatters.
        :type quivers : boolean

        :return : Boolean if it got executed successfully.
        """

        if dim_red is None:
            if self.bundle_tau:
                dim_red = self.tau_model
                print('We use the tau model from the BunDLeNet to project into 3D space.')
            else:
                dim_red = PCA(n_components=3)
                print('No mapping present. CREATING PCA MODEL ...')

        if not self._transform_points(dim_red):
            return False

        if interval is None:
            if self.data.fps is not None:
                print('The movie well be played in real time.')
                interval = 1000 / self.data.fps
            else:
                interval = 10
        self.interval = interval

        # We need to trim labs and colors if we have a Bundle
        if len(self.x) < len(self.colors):
            window = len(self.colors) - len(self.x)
            self.trimmed_colors = self.colors[window:]
        else:
            self.trimmed_colors = self.colors

        fig, self.movie_ax = plt.subplots(subplot_kw={'projection': '3d'})

        if quivers:
            self.movie_ax = self._add_quivers3D(self.movie_ax, self.x, self.y, self.z, colors=self.trimmed_colors)
        else:
            self.movie_ax.scatter(self.x, self.y, self.z, color=self.trimmed_colors, label=self.data.states, s=2,
                                  alpha=0.2)

        if grid_off:
            self.movie_ax.grid(False)
            self.movie_ax.set_axis_off()

        self.scatter = None
        self.animation = anim.FuncAnimation(fig, self._update, fargs=(grid_off, show_legend,), frames=len(self.x),
                                            interval=self.interval)
        plt.show()

        if save:
            name = str(input('What should the movie be called?')) + '.gif'
            self.save_gif(name)

        return True

    def _update(self, frame, grid_off, show_legend):
        if self.scatter is not None:
            self.scatter.remove()
        self.scatter = self.movie_ax.scatter(self.x[frame], self.y[frame], self.z[frame], s=20, alpha=0.8, color='red')
        self.movie_ax.set_title(f'Frame: {frame}\nBehavior: {self.data.states[self.data.B[frame]]}')

        if show_legend:
            legend_elements = self._generate_legend(self.data.B)
            self.movie_ax.legend(handles=legend_elements, loc='best')

        if grid_off:
            self.movie_ax.grid(False)
            self.movie_ax.set_axis_off()
        else:
            self.movie_ax.set_xlabel('Axes 1')
            self.movie_ax.set_ylabel('Axes 2')
            self.movie_ax.set_zlabel('Axes 3')

        return self.movie_ax

    def save_gif(self, name):
        if self.animation is None:
            print('No animation created yet.\nTo create one use \'.make_movie()\'.')
        else:
            print('This may take a while...')
            path = 'movies/' + name + '.gif'
            gif_writer = anim.PillowWriter(fps=int(1000 / self.interval), metadata=dict(artist='Me'), bitrate=1800)
            self.animation.save(path, writer=gif_writer, dpi=144)

    def behavioral_state_diagram(self, cog_stat_num=3, threshold=0.001, offset=2.5, save=True, show=True,
                                 adj_matrix=True):
        if self.data.p_markov is None:
            print('You need to run the behavioral probability trajectory clustering first (\'.cluster_BPT\').')
            return False
        # make the graph
        G = nx.DiGraph()
        node_colors = list(self.colordict.values()) * cog_stat_num

        T, states = adj_matrix_ncmcm(self.data.B, self.data, cog_stat_num=cog_stat_num)
        G.add_nodes_from(states)

        # adding edges
        for idx1, n1 in enumerate(states):
            for idx2, n2 in enumerate(states):
                if n1 != n2:
                    if T[idx1, idx2] > threshold:
                        G.add_edge(n1, n2, weight=T[idx1, idx2] * 1000)

        edge_colors = [node_colors[np.where(states == u)[0][0]] for u, v in G.edges()]

        node_sizes = np.diag(T) * 500 * (np.sqrt(T.shape[0]) / offset)

        mapping = {node: self.map_names(str(node)) for node in G.nodes()}

        print(G.nodes)
        G = nx.relabel_nodes(G, mapping)
        print(G.nodes)
        print(type(G))

        if adj_matrix:
            plt.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            plt.title('Adjacency Matrix Heatmap')
            plt.colorbar()
            plt.yticks(np.arange(T.shape[0]), G.nodes)
            plt.xlabel('Nodes')
            plt.ylabel('Nodes')
            plt.show()

        # pos = nx.circular_layout(G)

        c_nodes = []
        for c_num in range(cog_stat_num):
            c_nodes.append([n for n in G.nodes if n.split(':')[0] == 'C' + str(c_num + 1)])

        all_pos = []
        for c_node_group in c_nodes:
            all_pos.append(nx.circular_layout(G.subgraph(c_node_group)))

        adjusted_pos = {}
        degrees_list = np.linspace(0, 360, num=cog_stat_num, endpoint=False)
        print(f'DEGREES: {degrees_list}')
        for idx, p in enumerate(all_pos):
            adjusted_pos = circle_pos(p, adjusted_pos, degrees_list[idx], offset)

        # Plot graph
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw(G, adjusted_pos,
                with_labels=True,
                connectionstyle="arc3,rad=-0.2",
                node_color=node_colors,
                node_size=node_sizes,
                width=weights,
                arrows=True,
                arrowsize=10,
                edge_color=edge_colors)

        plt.title("Behavioral State Diagram")
        if save:
            name = str(input('File name for the plot? '))
            # plt.savefig(f'data/plots/state_diagram_worm_{self.data.data_set_no+1}.png', format='png')
            plt.savefig(f'data/plots/' + name + '.png', format='png')
        if show:
            plt.show()

        # net = Network()
        # net.from_nx(G)
        # net.show('test.html', notebook=False)

    def map_names(self, name):
        new_name = f'C{name[:-1]}:{self.data.states[int(name[-1])]}'
        return new_name

    def make_comparison(self, show_legend=False, dim_red=None):
        if dim_red is None:
            if self.bundle_tau:
                dim_red = self.tau_model
                print('We use the tau model from the BunDLeNet to project into 3D space.')
            else:
                dim_red = PCA(n_components=3)
                print('No mapping present. CREATING PCA MODEL ...')

        if not self._transform_points(dim_red):
            return False

        self._generate_label_counts()

        fig = plt.figure(figsize=(12, 8))

        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.scatter(self.x, self.y, self.z, color=self.trimmed_colors, s=2, alpha=0.5)  # label=self.data.states,
        ax1.set_title(f'True Label')

        ax2.scatter(self.x, self.y, self.z, color=self.diff_colors_pred, s=2,
                    alpha=0.5)  # color=self.diff_colors_pred, label=self.data.states,
        ax2.set_title(
            f'\nModel: {type(self.data.pred_model)}\nMapping: {type(dim_red)}\n\nAccuracy at {round(accuracy_score(self.data.B, self.data.B_pred), 2)}\n')

        ax3.scatter(self.x, self.y, self.z, color=self.colors_pred, s=2, alpha=0.5)  # label=self.data.states,
        ax3.set_title(f'Predicted Label')

        ax1.grid(False)
        ax1.set_axis_off()
        ax2.grid(False)
        ax2.set_axis_off()
        ax3.grid(False)
        ax3.set_axis_off()

        # plot the legend if wanted
        if show_legend:
            legend_elements1 = self._generate_legend(self.data.B)
            ax1.legend(title='True Labels', handles=legend_elements1, loc='upper center', bbox_to_anchor=(0.5, 0.))

            legend_elements2 = self._generate_legend(None, diff=True)
            ax2.legend(title='Wrongly Predicted Labels',handles=legend_elements2, loc='upper center', bbox_to_anchor=(0.5, 0.))

            legend_elements3 = self._generate_legend(self.data.B_pred)
            ax3.legend(title='Predicted Labels',handles=legend_elements3, loc='upper center', bbox_to_anchor=(0.5, 0.))

        fig.suptitle(f'{len(self.x)} Frames', fontsize='x-large', fontweight='bold')
        plt.show()


### FUNCTIONS MARKOV ###

def markovian(sequence, K=1000):
    P, states, M, N = compute_transition_matrix_lag2(sequence)

    # P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1])
    Pz0z1 = np.sum(P, axis=0)
    Pz1 = np.sum(P, axis=(2, 0))

    # P1 gives us the probability to be at x (=column) given that we came from at y (=row)
    tmp = Pz1.reshape(-1, 1)
    # It can happen that tmp contains zeros when a certain
    P1 = (Pz0z1 / tmp)

    if 0 in Pz1:
        print('HERE IS SOMETHING')
        print(np.unique(sequence), sequence)
        print('Pz1', Pz1)
        print('tmp', tmp)
        print('P1', P1)
        for idx in np.where(np.isnan(np.sum(P1, axis=1)))[0]:
            row_length = P1.shape[1]
            P1[idx, np.isnan(P1[idx])] = 1 / row_length

    # P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2])
    Pz1z2 = np.sum(P, axis=2)
    # I am replacing zeros in Pz1z2 with epsilon, so we do not encounter RuntimeWarnings
    epsilon = 1e-8
    Pz1z2 = np.where(Pz1z2 == 0, epsilon, Pz1z2)
    Pz1z2 = Pz1z2 / np.sum(Pz1z2)  # here I normalize it so the sum is 1 again

    P2 = P / np.tile(Pz1z2, (N, 1, 1))
    P2 = np.nan_to_num(P2)

    # Testing
    TH0 = np.zeros(K)
    for kperm in range(K):
        zH0, _ = simulate_markovian(M, P1)
        PH0 = np.zeros((N, N, N))

        for m in range(2, M):
            i = zH0[m]
            j = zH0[m - 1]
            k = zH0[m - 2]
            PH0[k, j, i] += 1

        PH0 = PH0 / (M - 2)
        Pz1z2H0 = np.sum(PH0, axis=2)

        # I am replacing zeros in Pz1z2H0 with epsilon, so we do not encounter RuntimeWarnings
        epsilon = 1e-8
        Pz1z2H0 = np.where(Pz1z2H0 == 0, epsilon, Pz1z2H0)
        Pz1z2H0 = Pz1z2H0 / np.sum(Pz1z2H0)  # here I normalize it so the sum is 1 again

        P2H0 = PH0 / np.tile(Pz1z2H0, (N, 1, 1))
        P2H0 = np.nan_to_num(P2H0)

        TH0[kperm] = sum(np.var(P2H0, axis=0).flatten())

    # compute p-value
    T = np.sum(np.var(P2, axis=2), axis=(0, 1))
    p = 1 - np.mean(T >= TH0)
    # I think the P1 should be returned since it is already the empirical transition matrix
    return p, P1


def compute_transition_matrix_lag2(sequence, normalize=True):
    states = sorted(np.unique(sequence))
    M = len(sequence)
    N = len(states)
    # sequence is translated into 0-N
    x = np.zeros(M, dtype=int)
    for i, state in enumerate(states):
        j = np.where([state == s for s in sequence])
        x[j] = i
    # tensor is created
    P = np.zeros((N, N, N))
    for m in range(2, M):
        # col
        i = x[m]
        # row
        j = x[m - 1]
        # depth
        k = x[m - 2]
        # from k to j to i
        P[k, j, i] += 1
    if normalize:
        P = P / (M - 2)
    return P, states, M, N


def simulate_markovian(M, P=np.array([]), N=1):
    if not len(P):
        P = np.random.rand(N, N)
        P = P / np.repeat(np.sum(P, axis=1)[np.newaxis, :], N, axis=0).T
    else:
        N = P.shape[0]

    # cumulative probabilities
    CP = np.cumsum(P, axis=1, dtype=float)
    # generate lots of data
    z = np.zeros(M, dtype=int)
    z[0] = np.random.randint(N)

    for m in range(1, M):
        prob = np.random.rand(1)
        # try:
        z[m] = np.where(CP[z[m - 1], :] >= prob)[0][0]
        # except Exception as e:
        #    print(f"Error occurred: {e}. Printing CP and prob for debugging:")
        #    print(f"P: {P}")
        #    print(f"CP: {CP}")
        #    print(f"prob: {prob}")
        #    exit()

    return z, P


def adj_matrix_ncmcm(B, data, cog_stat_num=3):
    """
    :param data: data from database
    :param B: all behaviors at each frame (e.g.: slow, rev ...)
    :param cog_stat_num: number of cognitive states in the plot (e.g.: C1, C2, C3 ...)
    :return:
    """
    xc = data.xc
    p = data.p_markov

    best_clustering_idx = np.argmax(p[cog_stat_num - 1, :])  # according to mr.markov himself
    best_clustering = xc[:, cog_stat_num - 1, best_clustering_idx].astype(int)
    cog_states = best_clustering

    b = np.unique(B)
    c = np.unique(cog_states)
    T = np.zeros((len(c) * len(b), len(c) * len(b)))

    states = [(cs + 1) * 10 + bs for cs in c for bs in b]

    for m in range(len(B) - 1):
        cur_sample = m
        next_sample = m + 1

        cur_state = np.where((cog_states[cur_sample] + 1) * 10 + B[cur_sample] == states)[0][0]
        next_state = np.where((cog_states[next_sample] + 1) * 10 + B[next_sample] == states)[0][0]
        T[next_state, cur_state] += 1

    # normalize T
    T = T / (len(B) - 1)
    T = T.T

    return T, states


def cart2pol(cartcord):
    theta = np.arctan2(cartcord[1], cartcord[0])
    rho = np.hypot(cartcord[0], cartcord[1])

    return theta, rho


def pol2cart(polcoord):
    x = polcoord[1] * np.cos(polcoord[0])
    y = polcoord[1] * np.sin(polcoord[0])

    return x, y


def average_markov_plot(markov_array):
    # Scatter plot each row with the index as x-values and the values as y-values
    for i in range(markov_array.shape[0]):
        plt.scatter(np.arange(markov_array.shape[1]), markov_array[i], label=f'Worm {i + 1}')

    mean_trendline = np.mean(markov_array, axis=0)
    plt.plot(np.arange(markov_array.shape[1]), mean_trendline, color='black', linestyle='--', label='Mean Trendline')

    # Add labels and legend
    plt.xlabel('Clusters/States')
    plt.ylabel('Probability')
    plt.axhline(0.05)
    plt.xticks(ticks=np.arange(0, markov_array.shape[1], 3), labels=np.arange(1, markov_array.shape[1] + 1, 3))
    plt.title('Markov Probability for Cognitive States')
    plt.legend()

    # Show the plot
    plt.show()


def circle_pos(old_pos, new_pos, degree, offset):
    for node, coords in old_pos.items():
        new_pos[node] = (coords[0] + offset * np.cos(np.radians(degree)),
                         coords[1] + offset * np.sin(np.radians(degree)))
    return new_pos


class CustomEnsembleModel():
    """
    This ensemble takes a model and creates binary predictors for each label-combination.
    As a prediction for each instance it gives the most abundant prediction from its sub-models.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.combinatorics = []
        self.ensemble_models = []

    def fit(self, neuron_traces, labels):
        self.ensemble_models = []
        self.combinatorics = list(combinations(np.unique(labels), 2))
        for idx, class_mapping in enumerate(self.combinatorics):
            b_model = clone(self.base_model)
            mapped_labels = np.array([label if label in class_mapping else -1 for label in labels])
            mask = mapped_labels != -1
            # apply mask to the dataset and only use instances of 'A' or 'B' to train
            neuron_traces_filtered = neuron_traces[mask]
            mapped_labels_filtered = mapped_labels[mask]
            b_model.fit(neuron_traces_filtered, mapped_labels_filtered)
            self.ensemble_models.append(b_model)
        return self

    def predict(self, neuron_traces):
        results = np.zeros((neuron_traces.shape[0], len(self.combinatorics))).astype(int)
        for idx, b_model in enumerate(self.ensemble_models):
            results[:, idx] = b_model.predict(neuron_traces)
        return [np.bincount(results[row, :]).argmax() for row in range(results.shape[0])]

    def predict_proba(self, neuron_traces):
        y_prob_map = np.zeros((neuron_traces.shape[0], len(self.combinatorics)))
        for idx, model in enumerate(self.ensemble_models):
            prob = model.predict_proba(neuron_traces)[:, 0]
            y_prob_map[:, idx] = prob
        return y_prob_map
