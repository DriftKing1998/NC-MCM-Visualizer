# import MyBundle
from scripts.functions import *
# from BunDLeNet import *
from scripts.MyBundle import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, List, Union
import mat73
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.animation as anim  # FuncAnimation

import networkx as nx


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
        self.yp_map = None
        self.xc = None

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
        vs = Visualizer(self.neuron_traces.T, self.B, xlabs=self.neuron_names, blabs=self.states, fps=self.fps)
        return vs

    def loadBundleVisualizer(self, l_dim=3):
        vs = Visualizer(self.neuron_traces, self.B, xlabs=self.neuron_names, blabs=self.states, fps=self.fps)
        time, newX = preprocess_data(vs.X.T, vs.fps)
        vs.X_, vs.B_ = prep_data(newX, vs.B, win=15)
        vs.model = BunDLeNet(latent_dim=l_dim)
        vs.model.build(input_shape=vs.X_.shape)
        vs.model.load_weights('data/generated/BunDLeNet_model_worm_' + str(self.data_set_no))
        vs.tau_model = vs.model.tau
        vs.bundle_tau = True
        return vs

    def fit_model(self, model, prob_map=True, markov_test=False, nrep=200, max_clusters=20, sim_markov=200):
        if not hasattr(model, 'fit'):
            print('Model has no method \'fit\'.')
            return None

        self.pred_model = model.fit(self.neuron_traces.T, self.B)
        self.B_pred = self.pred_model.predict(self.neuron_traces.T)
        print("Accuracy:", accuracy_score(self.B, self.B_pred))

        if prob_map:
            # get probabilities and weights
            self.yp_map = self.pred_model.predict_proba(self.neuron_traces.T)
            W = self.pred_model.coef_.T
            #ypall.append(yp)
            #Wall.append(W)

        if markov_test:
            self.p_markov = np.zeros((max_clusters, nrep))
            M = self.yp_map.shape[0]
            self.xc = np.zeros((M, max_clusters, nrep))

            for reps in range(nrep):
                print("Testing markovianity - repetition ", reps+1)
                for nrclusters in range(max_clusters):
                    # k-means
                    clusters = KMeans(n_clusters=nrclusters + 1, random_state=0, n_init="auto").fit(self.yp_map)
                    xctmp = clusters.labels_
                    p, _ = markovian(xctmp, K=sim_markov)
                    self.p_markov[nrclusters, reps] = p
                    self.xc[:, nrclusters, reps] = xctmp

        return self.pred_model

    def plot_markov(self):
        fig, ax = plt.subplots()
        data = self.p_markov[:, :].T
        print(data.shape)
        # Create boxplots
        ax.boxplot(data)
        ax.set_title(f'Probability of being a Markov process for worm {self.data_set_no+1}')
        ax.set_xlabel('Number of States/Clusters')
        ax.set_ylabel('Probability')
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()




class Visualizer():

    def __init__(self,
                 X: Union[List[List[float]], np.ndarray],
                 B: Union[List[int], List[str], np.ndarray],
                 xlabs: Optional[Union[List[str], np.ndarray]] = None,
                 blabs: Optional[Union[List[str], np.ndarray]] = None,
                 fps: float = None):
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

        # Input Preprocessing
        self.scatter = None
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
        self.bundle_tau = False
        self.model = None
        self.fps = fps
        self.animation = None
        self.interval = None

        # generate a color-dictionary for all states and generate the colors
        self.colordict = dict(zip(np.unique(self.B), generate_equidistant_colors(len(self.blabs))))
        self.colors = [self.colordict[val] for val in self.B]
        self.trimmed_colors = None

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

        im0 = ax.imshow(self.X.T, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None')
        # tell the colorbar to tick at integers
        # plt.colorbar(im0)
        ax.get_figure().colorbar(im0)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Neuronal activation")

        if show:
            plt.show()

    def _transform_points(self, dim_red):
        if dim_red is None:
            print('No mapping present. CREATING PCA MODEL ...')
            pca = PCA(n_components=3)
            dim_red = pca
            transformed_points = dim_red.fit_transform(self.X)
        else:
            if self.bundle_tau:  # isinstance(self.model, BunDLeNet):
                print('HAVE BUNDLE MODEL')
                # self.tau_model = self.model.tau
                transformed_points = np.asarray(self.tau_model(self.X_[:, 0]))

            else:
                if hasattr(dim_red, 'fit_transform'):
                    if dim_red.get_params()['n_components'] == 3:
                        print('HAVE different mapping MODEL')
                        if isinstance(dim_red, NMF):
                            scaler = MinMaxScaler(feature_range=(0, np.max(self.X)))
                            X_scaled = scaler.fit_transform(self.X)
                            transformed_points = dim_red.fit_transform(X_scaled)
                        else:
                            transformed_points = dim_red.fit_transform(self.X)
                    else:
                        print('The selected model does not project to a 3 dimensional space.')
                        return False
                else:
                    print('The selected model has no attribute \'fit_transform\'. (SKLEARN models are recommended)')
                    return False

        print(transformed_points.T.shape)
        self.x, self.y, self.z = transformed_points.T
        return True

    def plot3D_mapping(self, dim_red=None, show_legend=False, grid_off=True, quivers=False):
        if dim_red is None:
            dim_red = self.tau_model
            print('The current latent dimension mapping (tau) is used for plotting.')

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
            ax = self._add_quivers(ax, colors)
        else:
            ax.scatter(self.x, self.y, self.z, label=self.blabs, s=1, alpha=0.5, color=colors)

        ax.set_xlabel('Axes 1')
        ax.set_ylabel('Axes 2')
        ax.set_zlabel('Axes 3')

        # plot the legend if wanted
        if show_legend:
            legend_elements = self._generate_legend(self.blabs)
            ax.legend(handles=legend_elements)

        if grid_off:
            ax.grid(False)
            ax.set_axis_off()

        plt.show()

    def _generate_legend(self, classifier, diff=False):
        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colordict[idx],
                                      markersize=10, label=r'$\mathbf{' + lab + '}$' + f' ({list(self.B).count(idx)})')
                           for idx, lab in enumerate(self.blabs)]

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

        return legend_elements

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
        time, newX = preprocess_data(self.X, self.fps)
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

    def _add_quivers(self, ax, colors):
        dx = np.diff(self.x)  # Differences between x coordinates
        dy = np.diff(self.y)  # Differences between y coordinates
        dz = np.diff(self.z)  # Differences between z coordinates
        lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        for idx in range(len(dx)):
            ax.quiver(self.x[idx], self.y[idx], self.z[idx], dx[idx], dy[idx], dz[idx], color=colors[idx],
                      arrow_length_ratio=0.1 / lengths[idx], alpha=0.5, linewidths=0.5)
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
            dim_red = self.tau_model
            print('Try using current latent dimension mapping (tau) for plotting.')

        if not self._transform_points(dim_red):
            return False

        if interval is None:
            print('The movie well be played in real time.')
            interval = 1000 / self.fps
        self.interval = interval

        # We need to trim labs and colors if we have a Bundle
        if len(self.x) < len(self.colors):
            window = len(self.colors) - len(self.x)
            self.trimmed_colors = self.colors[window:]
        else:
            self.trimmed_colors = self.colors

        fig, self.movie_ax = plt.subplots(subplot_kw={'projection': '3d'})

        if quivers:
            self.movie_ax = self._add_quivers(self.movie_ax, self.trimmed_colors)
        else:
            self.movie_ax.scatter(self.x, self.y, self.z, color=self.trimmed_colors, label=self.blabs, s=2, alpha=0.2)

        if grid_off:
            self.movie_ax.grid(False)
            self.movie_ax.set_axis_off()

        self.scatter = None
        self.animation = anim.FuncAnimation(fig, self._update, fargs=(grid_off, show_legend,), frames=len(self.x),
                                            interval=self.interval)
        plt.show()
        return True

    def _update(self, frame, grid_off, show_legend):
        if self.scatter is not None:
            self.scatter.remove()
        self.scatter = self.movie_ax.scatter(self.x[frame], self.y[frame], self.z[frame], s=20, alpha=0.8, color='red')
        self.movie_ax.set_title(f'Frame: {frame}\nBehavior: {self.blabs[self.B[frame]]}')

        if show_legend:
            legend_elements = self._generate_legend(self.blabs)
            self.movie_ax.legend(handles=legend_elements)

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

    def behavioral_state_diagram(self):
        # make the graph
        G = nx.DiGraph()
        G.add_nodes_from(self.blabs)
        node_colors = list(self.colordict.values())

        cog_states = [1] * (len(self.B))
        T = adj_matrix_ncmcm(self.B, cog_states)

        # adding edges
        for n1 in range(len(self.blabs)):
            for n2 in range(len(self.blabs)):
                if n1 != n2:
                    G.add_edge(self.blabs[n1], self.blabs[n2], weight=T[n1, n2] * 1000, color=node_colors[n1])

        node_sizes = np.diag(T) * 1000

        pos = nx.circular_layout(G)

        # Plot graph
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw(G, pos,
                with_labels=True,
                connectionstyle="arc3,rad=-0.2",
                node_color=node_colors,
                node_size=node_sizes,
                width=weights,
                arrows=True,
                arrowsize=10)
        # nx.draw(G, pos, with_labels=True, arrows=True)
        plt.title("Behavioral State Diagram")
        plt.show()

    def make_comparison(self, show_legend=False):
        if not hasattr(self, 'true_label_counts'):
            self._generate_label_counts()

        fig = plt.figure(figsize=(12, 5))

        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.scatter(self.x, self.y, self.z, color=self.colors, label=self.ylabs, s=2, alpha=0.5)
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        ax1.set_zlabel('PC 3')
        ax1.set_title(f'True Label')

        ax2.scatter(self.x, self.y, self.z, color=self.diff_colors_pred, label=self.ylabs, s=2, alpha=0.5)
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.set_zlabel('PC 3')
        ax2.set_title(
            f'Model: {type(self.untrained_model)}\n\nAccuracy at {round(accuracy_score(self.Y, self.Y_pred), 2)}\n')

        ax3.scatter(self.x, self.y, self.z, color=self.colors_pred, label=self.ylabs, s=2, alpha=0.5)
        ax3.set_xlabel('PC 1')
        ax3.set_ylabel('PC 2')
        ax3.set_zlabel('PC 3')
        ax3.set_title(f'Predicted Label')

        # plot the legend if wanted
        if show_legend:
            pred_legend_elements, pred_y_labels = self._generate_legend(self.pred_label_counts)
            l1 = fig.legend(pred_legend_elements, pred_y_labels, ncol=2, loc='lower right', fontsize='x-small')
            l1.set_title('Label - (wrong/correct)', prop={'weight': 'bold'})

            true_legend_elements, true_y_labels = self._generate_legend(self.true_label_counts)
            l2 = fig.legend(true_legend_elements, true_y_labels, ncol=2, loc='lower left', fontsize='x-small')
            l2.set_title('Label - (wrong/correct)', prop={'weight': 'bold'})

            diff_legend_elements, diff_y_labels = self._generate_legend(self.diff_label_counts, True)
            l3 = fig.legend(diff_legend_elements, diff_y_labels, ncol=2, loc='lower center', fontsize='x-small')
            l3.set_title(f'predicted incorrect as (amount)', prop={'weight': 'bold'})

        fig.suptitle(f'{len(self.x)} Frames', fontsize='x-large', fontweight='bold')
        plt.show()


### FUNCTIONS MARKOV ###

def markovian(sequence, K=1000):
    P, states, M, N = compute_transition_matrix_lag2(sequence)

    # P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1])
    Pz0z1 = np.sum(P, axis=0)
    Pz1 = np.sum(P, axis=(2, 0))
    # This gives us the probability to be at x (=column) given that we came from at y (=row)
    P1 = (Pz0z1 / Pz1.reshape(-1, 1))

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
        z[m] = np.where(CP[z[m - 1], :] >= prob)[0][0]

    return z, P


def adj_matrix_ncmcm(B, cog_states):
    T = np.zeros((8, 8))
    b = np.unique(B)
    c = np.unique(cog_states)
    states = [cs * 10 + bs for cs in c for bs in b]

    for m in range(len(B) - 1):
        cur_sample = m
        next_sample = m + 1

        cur_state = np.where(cog_states[cur_sample] * 10 + B[cur_sample] == states)[0][0]
        next_state = np.where(cog_states[next_sample] * 10 + B[next_sample] == states)[0][0]
        T[next_state, cur_state] += 1

    # normalize T
    T = T / (len(B) - 1)
    T = T.T
    return T


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
    plt.xticks(ticks=np.arange(0, markov_array.shape[1], 3), labels=np.arange(1, markov_array.shape[1]+1, 3))
    plt.title('Markov Probability for Cognitive States')
    plt.legend()

    # Show the plot
    plt.show()
