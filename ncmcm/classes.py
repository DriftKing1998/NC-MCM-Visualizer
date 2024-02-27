from ncmcm.functions import *
from ncmcm.BundDLeNet import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA, NMF
from typing import Optional, List, Union
import mat73
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.animation as anim  # FuncAnimation
from sklearn.base import clone
import networkx as nx
from itertools import combinations
from pyvis.network import Network


class Loader:

    def __init__(self,
                 data_set_no):
        """
        Reads in the data from the all files corresponding to the selected dataset.
        It stores all values into numpy arrays.

        :param data_set_no: Defines which CSV files will be read.
        :type data_set_no: int
        """

        self.data_set_no = data_set_no
        data_dict = mat73.loadmat('/data/NoStim_Data.mat')
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

        self.data = self.neuron_traces, self.B, self.neuron_names, self.states, self.fps


class Database:

    def __init__(self,
                 neuron_traces: Union[List[List[float]], np.ndarray],
                 behavior: Union[List[int], List[str], np.ndarray],
                 neuron_names: Optional[Union[List[str], np.ndarray]] = None,
                 states: Optional[Union[List[str], np.ndarray]] = None,
                 fps: float = None,
                 name: str = 'nc-mcm'):

        """
        Reads in the data preferably as numpy arrays or Python-lists. If the behavior array consists of strings, a
        list for translation is created (states) and the behavior is transformed into integers (index of states). Also
        creates colors for plotting. The object can be seen as a container for all the data which is used to plot later.

        :param neuron_traces: Defines which CSV files will be read.
        :type neuron_traces: float

        :param behavior: Defines which CSV files will be read.
        :type behavior: int

        :param neuron_names: Defines which CSV files will be read.
        :type neuron_names: int

        :param states: Defines which CSV files will be read.
        :type states: int

        :param fps: Defines which CSV files will be read.
        :type fps: int

        :param name: Separator to split the CSV files.
        :type name: string
        """

        self.neuron_traces = np.asarray(neuron_traces)
        self.fps = fps
        self.name = name
        behavior = np.asarray(behavior)
        # If B is not an integer array we have to transform it into one
        if not (np.issubdtype(behavior.dtype, int) or np.issubdtype(behavior.dtype, np.integer)):
            print('B has to be transformed into a integer-array. Translation is accessed by\'self.states\'.')
            newB, blabs = make_integer_list(behavior)
            self.B = np.asarray(newB)
            self.states = np.asarray(blabs).astype(str)
        else:
            self.B = behavior
            self.states = states
        # If no Neuron Names are given, they are generated
        if neuron_names is None:
            self.neuron_names = np.asarray(range(self.neuron_traces.shape[0])).astype(str)
        else:
            self.neuron_names = np.asarray(neuron_names)

        # If no State Names are given, they are generated
        if states is None:
            print('State-names are created from behavior-labels. Translation is accessed by\'self.states\'.')
            newB, blabs = make_integer_list(self.B)
            self.states = np.asarray(blabs).astype(str)
        else:
            self.states = np.asarray(states)

        self.pred_model = None
        self.B_pred = None
        self.yp_map = None
        self.p_memoryless = None
        self.p_stationary = None
        self.xc = None

        self.colordict = dict(zip(np.unique(self.B), generate_equidistant_colors(len(self.states))))
        self.colors = [self.colordict[val] for val in self.B]

        if self.B.shape[0] != self.neuron_traces.shape[1] or self.neuron_names.shape[0] != self.neuron_traces.shape[0]:
            print('Error! Data seems not to fit together.')
            print(f'Frames of behavior {self.B.shape[0]} are not the same length as frames of neuronal data: '
                  f'{self.neuron_traces.shape[1]}')
            print(f'Or Neuron-names {self.neuron_names.shape[0]} are not the same length as neurons were recorded '
                  f'{self.neuron_traces.shape[0]}')

    def exclude_neurons(self,
                        exclude_neurons):
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

    def createVisualizer(self,
                         mapping=None,
                         l_dim=3,
                         epochs=2000,
                         window=15,
                         use_predictor=True):
        """
        Takes either a mapping to visualize the data (e.g.: PCA) or parameters for a BundDLeNet (l_dim, epochs, window)
        which will be used to visualize the data. If a BundDLeNet is created, it will be used to predict behaviors in
        future plots. Otherwise, the model fitted on the Database-object, will be used as a predictor, if it exists.

        :param mapping: A mapping (such as PCA) which is used for the projection into three dimension. It needs to have
         the method 'fit_transform'.

        :param l_dim: Latent dimension the BundDLeNet maps to (for visualisation: 3D; For further use: XD)
        :type l_dim: int

        :param epochs: Epochs for the BundDLeNet
        :type epochs: int

        :param window: Window-size for BundDLeNet
        :type window: int

        :param use_predictor: If the BundDLeNet Predictor should be used as prediction model
        :type use_predictor: bool

        return: Will return the correctly configured Visualizer object or None
        :type: Visualizer
        """

        # If a mapping is provided
        if mapping is not None:
            vs = Visualizer(self, mapping)
        # otherwise a BundDLeNet is created
        else:
            if self.fps is None:
                print('Give \'self.fps\' a value (float) first!')
                return None
            time, newX = preprocess_data(self.neuron_traces.T, self.fps)
            X_, B_ = prep_data(newX, self.B, win=window)
            model = BundDLeNet(latent_dim=l_dim, behaviors=len(self.states))
            model.build(input_shape=X_.shape)

            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
            loss_array = train_model(
                X_,
                B_,
                model,
                optimizer,
                gamma=0.9,
                n_epochs=epochs
            )

            vs = Visualizer(self, model.tau, transform=False)
            vs.X_ = X_
            vs.B_ = B_
            vs.model = model
            vs.loss_array = loss_array
            vs.tau_model = model.tau
            vs.bn_tau = True
            # I need to do this later, since X_ is not defined yet
            vs._transform_points(vs.mapping)
            if use_predictor:
                vs.useBundDLePredictor()

        return vs

    def loadBundDLeVisualizer(self,
                              weights_path=None,
                              l_dim=3,
                              window=15,
                              use_predictor=True):
        """
        Takes a path to the weights (standard directory = /data/generated/BundDLeNet_model_<name>) and parameters for a
        BundDLeNet (l_dim, window) which will be created and the weights will be laoded in. One can also choose if the
        predictor of the BundDLeNet or the model form the Database-object will be used (if present).

        :param weights_path: A path to the "directory + name of the weights file"
        :type weights_path: string

        :param l_dim: Latent dimension the BundDLeNet maps to (for visualisation: 3D; For further use: XD)
        :type l_dim: int

        :param window: Window-size for BundDLeNet
        :type window: int

        :param use_predictor: Boolean if the BundDLeNet Predictor should be used for plots.
        :type use_predictor: bool

        :return: Will return the correctly configured Visualizer object or None
        :type: Visualizer
        """

        time, newX = preprocess_data(self.neuron_traces.T, self.fps)
        X_, B_ = prep_data(newX, self.B, win=window)
        model = BundDLeNet(latent_dim=l_dim, behaviors=len(self.states))
        model.build(input_shape=X_.shape)

        try:
            if weights_path is None:
                model.load_weights('data/generated/BundDLeNet_model_' + self.name)
            else:
                model.load_weights(weights_path)
        except Exception as e:
            print(f'Error {e}! No such file.')
            return None

        vs = Visualizer(self, model.tau, transform=False)
        vs.X_ = X_
        vs.B_ = B_
        vs.model = model
        vs.tau_model = model.tau
        vs.bn_tau = True
        # I need to do this later, since X_ is not defined yet
        vs._transform_points(vs.mapping)
        if use_predictor:
            vs.useBundDLePredictor()

        return vs

    # Maybe make a 90/10 training test set split.
    def fit_model(self,
                  base_model,
                  prob_map=True,
                  binary=True):
        """
        Allows to fit a model which is used to predict behaviors from the neuron traces (accuracy is printed). Its
        probabilities are used for an eventual clustering.

        :param base_model:

        :param prob_map: A boolean indicating if a probability map is created for each frame. This is used in the
        behavioral probability trajectory clustering (.cluster_BPT()).

        :param binary: A boolean indicating if the CustomEnsembleModel should be created. It makes a set of models for
        each behavior, to differentiate it from each of the others behaviors.

        :return: Boolean success indicator
        """
        if not hasattr(base_model, 'fit'):
            print('Model has no method \'fit\'.')
            return False

        # For a binary regression by hand
        if binary:
            self.pred_model = CustomEnsembleModel(base_model).fit(self.neuron_traces.T, self.B)
        # For a multiclass regression
        else:
            self.pred_model = base_model.fit(self.neuron_traces.T, self.B)

        self.B_pred = np.asarray(self.pred_model.predict(self.neuron_traces.T))
        print("Accuracy:", accuracy_score(self.B, self.B_pred))
        if prob_map:
            # get probabilities and weights
            self.yp_map = self.pred_model.predict_proba(self.neuron_traces.T)
            print(f'Probability map has shape: {self.yp_map.shape}')
        return True

    def cluster_BPT(self,
                    nrep=200,
                    max_clusters=20,
                    sim_m=500,
                    sim_s=500,
                    chunks=7,
                    clustering='kmeans',
                    kmeans_init='auto',
                    plot_markov=True,
                    stationary=False):
        """
        Clusters behavioral probability trajectories if a model has been fitted on the data.

        :param nrep: Repetitions of repeated clustering for each number of clusters.
        :type nrep: int

        :param max_clusters: Maximal amount of clusters - 1 to "max_clusters" clusters will be done.
        :type max_clusters: int

        :param sim_m: Amount of simulations will be done for the test statistics of each memory-less test
        :type sim_m: int

        :param sim_s: Amount of simulations will be done for the test statistics of each stationary test
        :type sim_s: int

        :param chunks: Amount of chunks created for the stationary test (frobenius norm of the chunks is used)
        :type chunks: int

        :param kmeans_init: Is for the "n_init" parameter of KMeans and defines how often K-means is initialized after
        clustering (the best result is picked by the sklearn package)
        :type kmeans_init: string or int

        :param plot_markov: If the result should be plotted
        :type plot_markov: bool

        :param stationary: If stationarity test should also be done
        :type stationary: bool

        :return: Boolean success indicator
        """
        if self.yp_map is None:
            print(f'You first need to fit a model (eg. Logistic Regression), '
                  f'which will be used to map to behavioral probability trajectories.\n'
                  f'Use \'.fit_model(<your_model>)\' on this instance before.')
            return False

        M = self.yp_map.shape[0]
        self.p_memoryless = np.zeros((max_clusters, nrep))
        self.p_stationary = np.zeros((max_clusters, nrep))
        self.xc = np.zeros((M, max_clusters, nrep))

        for reps in range(nrep):
            print("Testing markovianity - repetition ", reps + 1)
            for nrclusters in range(max_clusters):
                # Clustering in probability space
                if clustering is 'kmeans':
                    clusters = KMeans(n_clusters=nrclusters + 1, n_init=kmeans_init).fit(self.yp_map)
                    xctmp = clusters.labels_
                elif clustering is 'spectral':
                    clusters = SpectralClustering(n_clusters=nrclusters + 1).fit(self.yp_map)
                    xctmp = clusters.row_labels_
                else:
                    raise ValueError("Invalid value for 'clustering' parameter. "
                                     "It should be either 'kmeans' or 'spectral'. ")

                p, _ = markovian(xctmp, sim_memoryless=sim_m)
                self.p_memoryless[nrclusters, reps] = p

                if stationary:
                    _, p_adj_s = test_stationarity(xctmp, parts=chunks, plot=False, sim_stationary=sim_s)
                    self.p_stationary[nrclusters, reps] = p_adj_s

                self.xc[:, nrclusters, reps] = xctmp

        if plot_markov:
            self._plot_markov(stationary)
        return True

    def _plot_markov(self, stationary=False):
        """
        Creates the markovian plot.
        """
        fig, ax = plt.subplots()
        # Plotting Memorylessness
        data_m = self.p_memoryless[:, :].T
        boxplot_m = ax.boxplot(data_m, patch_artist=True, boxprops=dict(facecolor='lightblue', edgecolor='blue'))
        box_label_m = 'Memoryless'
        boxplot_m['boxes'][0].set_label(box_label_m)
        boxplot_m['boxes'][0].set_label(box_label_m)
        # Plotting Stationarity if wanted
        if stationary:
            data_s = self.p_stationary[:, :].T
            boxplot_s = ax.boxplot(data_s, patch_artist=True, boxprops=dict(facecolor='salmon', edgecolor='red'))
            box_label_s = 'not Stationary'
            boxplot_s['boxes'][0].set_label(box_label_s)

        ax.set_title(f'Probability of being a Markov process for {self.name}')
        ax.set_xlabel('Number of States/Clusters')
        ax.set_ylabel('Probability')
        ax.axhline(0.05)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        return True

    def step_plot(self,
                  clusters=5,
                  nrep=10,
                  sim_m=300,
                  sim_s=300,
                  save=False,
                  show=True,
                  png_name=None):
        """
        Creates a plot consisting of 4 plots. The first one showing behavioral labels plotted onto the 2 principal
        components of the neuronal data, the second one showing behavioral labels plotted onto behavioral probability
        trajectories, the third one showing cognitive labels plotted onto behavioral probability trajectories and the
        last one showing cognitive labels plotted onto the 2 principal components of the neuronal data.

        :param clusters: Amount of cognitive states (clusters) to use.
        :type clusters: int

        :param nrep: Repetitions of repeated clustering for each number of clusters.
        :type nrep: int

        :param sim_m: Amount of simulations will be done for the test statistics of each memory-less test
        :type sim_m: int

        :param sim_s: Amount of simulations will be done for the test statistics of each stationary test
        :type sim_s: int

        :param png_name: Name of the plot if it should be saved, otherwise it will be named step_plot_<self.name>Mouse.png
        :type png_name: string

        :param save: Boolean if it should be saved
        :param show: Boolean if it should be shown

        :return: Boolean success indicator
        """
        if self.p_memoryless is None or self.p_memoryless.shape[0] < clusters:
            print('There were no BPT-clusterings computed. It will be done now...')
            self.fit_model(LogisticRegression(solver='lbfgs', max_iter=1000), binary=True)
            self.cluster_BPT(nrep=nrep, max_clusters=clusters, sim_m=sim_m, sim_s=sim_s, plot_markov=False)

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
        best_clustering_idx = np.argmax(self.p_memoryless[clusters - 1, :])  # according to mr.markov himself
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
        fig.suptitle(f'{self.name} with {clusters} cognitive states')
        if save:
            if png_name:
                plt.savefig(f'data/plots/{png_name}.png', format='png')
            else:
                plt.savefig(f'data/plots/step_plot_{self.name}.png', format='png')
        if show:
            plt.show()
        return True

    def _add_quivers2D(self, ax, x, y, colors=None):
        """
        Function to add 2D quivers of correct size to an axis.
        """
        if colors is None:
            colors = self.colors[:-1]
        dx = np.diff(x)  # Differences between x coordinates
        dy = np.diff(y)  # Differences between y coordinates
        ax.quiver(x[:-1], y[:-1], dx, dy, color=colors, scale_units='xy', angles='xy', scale=1, alpha=0.5)
        return ax

    def behavioral_state_diagram(self,
                                 cog_stat_num=3,
                                 threshold=None,
                                 offset=2.5,
                                 adj_matrix=True,
                                 show=True,
                                 save=True,
                                 interactive=False,
                                 clustering_rep=None):
        """
        Creates a behavioral state diagram using the defined states as a directed graph.

        :param cog_stat_num: Defines the amount of cognitive states used.
        :type cog_stat_num: int

        :param threshold: A threshold which is used to display edges in the graph (smaller values are not plotted)
        :type threshold: float

        :param offset: Distance between clusters
        :type offset: float

        :param clustering_rep: If a certain clustering should be used
        :type clustering_rep: None or int

        :param adj_matrix: If the adjacency matrix should be plotted
        :param show: If the matplotlib plot should be shown
        :param save: If the matplotlib plot should be saved
        :param interactive: If the html plot should be saved

        :return: Boolean success indicator
        """
        if self.p_memoryless is None or self.p_memoryless.shape[0] < cog_stat_num:
            print('You need to run the behavioral probability trajectory clustering first (\'.cluster_BPT\').')
            return False
        if threshold is None:
            threshold = 1 / (500 * cog_stat_num)
            print('Calcualted threshold is: ', threshold)
        # make the graph
        G = nx.DiGraph()
        node_colors = list(self.colordict.values()) * cog_stat_num

        T, cog_beh_states = adj_matrix_ncmcm(self, cog_stat_num=cog_stat_num, clustering_rep=clustering_rep)
        G.add_nodes_from(cog_beh_states)

        # adding edges
        for idx1, n1 in enumerate(cog_beh_states):
            for idx2, n2 in enumerate(cog_beh_states):
                if n1 != n2:
                    if T[idx1, idx2] > threshold:
                        G.add_edge(n1, n2, weight=T[idx1, idx2] * 1000)

        edge_colors = [node_colors[np.where(cog_beh_states == u)[0][0]] for u, v in G.edges()]
        node_sizes = np.diag(T) * 500 * (np.sqrt(T.shape[0]) / offset)
        mapping = {node: self.map_names(str(node)) for node in G.nodes()}

        G = nx.relabel_nodes(G, mapping)

        if adj_matrix:
            fig, ax = plt.subplots(1, 2)
            ax_a = ax[0]
            ax_g = ax[1]
            ax_a.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax_a.title('Adjacency Matrix Heatmap')
            ax_a.colorbar()
            ax_a.yticks(np.arange(T.shape[0]), G.nodes)
            ax_a.xlabel('Nodes')
            ax_a.ylabel('Nodes')
            #plt.show()
        else:
            fig, ax_g = plt.subplots()


        cog_groups = []
        for c_num in range(cog_stat_num):
            cog_groups.append([n for n in G.nodes if n.split(':')[0] == 'C' + str(c_num + 1)])

        all_pos = []
        for c_node_group in cog_groups:
            all_pos.append(nx.circular_layout(G.subgraph(c_node_group)))

        adjusted_pos = {}
        degrees_list = np.linspace(0, 360, num=cog_stat_num, endpoint=False)
        for idx, current_pos in enumerate(all_pos):
            adjusted_pos = shift_pos_by(current_pos, adjusted_pos, degrees_list[idx], offset)

        # Plot graph
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        if show:
            nx.draw(G, adjusted_pos,
                    with_labels=True,
                    connectionstyle="arc3,rad=-0.2",
                    node_color=node_colors,
                    node_size=node_sizes,
                    width=weights,
                    arrows=True,
                    arrowsize=10,
                    edge_color=edge_colors,
                    ax=ax_g)
            plt.title("Behavioral State Diagram")
            plt.show()

        if save:
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
            name = str(input('File name for the plot? '))
            plt.savefig(f'data/plots/{name}.png', format='png')
            print(f'Plot has been saved under: data/plots/{name}.png')
            plt.close()
        # This right here will create the interactive HTML plot
        if interactive:
            net = Network(directed=True  # ,
                          # select_menu=True,  # Show part 1 in the plot (optional)
                          )
            net.from_nx(G)
            for node in net.nodes:
                c, b = node['id'].split(':')
                c_int = int(c[1:]) - 1
                b_int = np.where(np.asarray(self.states) == b)[0][0]
                r, g, b = self.colordict[b_int]
                node['color'] = f'rgb({r * 255},{g * 255},{b * 255})'
                node['size'] = np.sqrt(node_sizes[c_int * (len(self.states)) + b_int])

            name = str(input('File name for the html-plot? '))
            net.show(f'data/plots/{name}.html', notebook=False)
            print(f'Plot has been saved under: data/plots/{name}.html')
        return True

    def map_names(self,
                  name):
        """
        Used to generate a state-name from a number
        """
        new_name = f'C{name[:-2]}:{self.states[int(name[-2:])]}'
        return new_name

    def plotting_neuronal_behavioural(self,
                                      vmin=0,
                                      vmax=2):
        """
        Plots neuronal data and behavioral data as a timeseries.
        :param vmin: minimal value for neuronal data values
        :param vmax: maximal value for neuronal data values
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 4))
        self._neurons(ax=axs[0], vmin=vmin, vmax=vmax)
        self._behavior(ax=axs[1])
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def _behavior(self,
                  ax=None):
        """
        Plots behavioral data as a timeseries onto an axis if given one. Otherwise, a figure will be created and shown.
        """
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        cmap = plt.get_cmap('Pastel1', np.max(self.B) - np.min(self.B) + 1)
        im1 = ax.imshow([self.B], cmap=cmap, vmin=np.min(self.B) - 0.5, vmax=np.max(self.B) + 0.5,
                        aspect='auto')
        # tell the colorbar to tick at integers
        cax = ax.get_figure().colorbar(im1, ticks=np.arange(np.min(self.B), np.max(self.B) + 1))
        if len(np.unique(self.B)) == len(self.states):
            cax.ax.set_yticklabels(self.states)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Behaviour")
        ax.set_yticks([])
        # ax.set_title(f'Sample no#{self.data_set_no + 1}')

        if show:
            plt.show()

    def _neurons(self,
                 ax=None,
                 vmin=0,
                 vmax=2):
        """
        Plots neuronal data as a timeseries onto an axis if given one. Otherwise, a figure will be created and shown.
        :param vmin: minimal value for neuronal data values
        :param vmax: maximal value for neuronal data values
        """
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(10, 2))

        im0 = ax.imshow(self.neuron_traces, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None')
        # tell the colorbar to tick at integers
        # plt.colorbar(im0)
        ax.get_figure().colorbar(im0)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Neurons")
        ax.set_title("Neuronal activation")
        if show:
            plt.show()


class Visualizer():

    def __init__(self,
                 Data: Database,
                 mapping,
                 transform=True):
        """
        Creates a Visualizer object using a Database object and a mapping.

        :param Data: Database object with data to be plotted later
        :type Data: Database

        :param mapping: Some sort of dimensionality reduction, preferably to 3D space, so plotting is possible.

        :param transform: Boolean if points should be transformed directly using the mapping. Normally "True" is fine.
        """

        # Setting Attributes
        self.data = Data
        self.B_pred = self.data.B_pred
        self.pred_model = self.data.pred_model
        self.X_ = None
        self.B_ = None
        # Mapping
        self.mapping = mapping
        if transform:
            self._transform_points(self.mapping)
        else:
            self.transformed_points = None

        # Colors for plotting
        self.plot_colors = None
        self.colors_diff_pred = None
        self.colors_pred = None
        # BundDLeNet
        self.loss_array = None
        self.tau_model = None
        self.bn_tau = False
        self.model = None
        # Animation
        self.animation = None
        self.interval = None
        self.scatter = None

    def change_mapping(self,
                       new_mapping):
        """
        If a different mapping should be used in the future.

        :param new_mapping: Some sort of dimensionality reduction, preferably to 3D space, so plotting is possible.

        :return: Boolean success indicator
        """
        self.bn_tau = False
        if self._transform_points(new_mapping):
            self.mapping = new_mapping
            return True
        else:
            print('Mapping was not changed.')
            return False

    ### DIAGNOSTICS ###
    def plot_mapping(self,
                     show_legend=False,
                     grid_off=True,
                     quivers=False,
                     show=True,
                     draw=True):
        """
        Uses the mapping of the Visualizer to plot the datapoints into 3D space

        :param show_legend: If the legend should be shown

        :param grid_off: If the grid should be shown

        :param quivers: If quivers should be used, otherwise a scatterplot is created

        :param show: If the plot should be shown, otherwise the plot's components will be returned

        :return: Either a boolean success indicator or figure, axis and legend handles
        """
        if self.transformed_points.shape[0] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # We need to trim labs and colors if we have a BundDLe
        if self.transformed_points.shape[1] < len(self.data.colors):
            window = len(self.data.colors) - self.transformed_points.shape[1]
            self.plot_colors = self.data.colors[window:]
        else:
            self.plot_colors = self.data.colors

        if quivers:
            ax = self._add_quivers3D(ax, *self.transformed_points, colors=self.plot_colors, draw=draw)
        else:
            ax.scatter(*self.transformed_points, label=self.data.states, color=self.plot_colors, s=1, alpha=draw)

        # plot the legend if wanted
        if show_legend:
            legend_elements = self._generate_legend(self.data.B)
            ax.legend(handles=legend_elements, fontsize='small', loc='lower center', bbox_to_anchor=[1,0])
        else:
            legend_elements = False

        if grid_off:
            ax.grid(False)
            ax.set_axis_off()
        else:
            ax.set_xlabel('Axes 1')
            ax.set_ylabel('Axes 2')
            ax.set_zlabel('Axes 3')

        if show:
            ax.set_title(f'Mapping: {type(self.mapping)}')
            plt.show()
            return True
        else:
            return fig, ax, legend_elements

    def _transform_points(self,
                          mapping):
        """
        Uses the mapping given to transform the data points (Database.neuron_traces). Also checks if the mapping is a
        neural network or any other sklearn dimensionality reduction.

        :param mapping: Dimensionality reduction mapping for the data points (for visualization mapping to 3D space)
        :type mapping: neural network or any sklearn-object with method "fit_transform"

        :return: Boolean success indicator
        """
        if mapping is None:  # This should not happen normally
            print('No mapping present. CREATING PCA MODEL ...')
            mapping = PCA(n_components=3)
            transformed_points = mapping.fit_transform(self.data.neuron_traces.T)
        # If we are using the TAU model to map into 3D space
        elif isinstance(mapping, tf.keras.Sequential):
            # If the mapping is BundDLeNet we use the 'windowed' input
            if self.X_ is not None and mapping.input_shape[1] == self.X_.shape[2]:
                self.bn_tau = True
                transformed_points = np.asarray(mapping(self.X_[:, 0]))
            else:
                transformed_points = np.asarray(mapping(self.data.neuron_traces.T))
        # This happens if we give some mapping which is not a NN
        elif hasattr(mapping, 'fit_transform'):
            if mapping.get_params()['n_components'] == 3:
                # print('HAVE mapping MODEL')
                if isinstance(mapping, NMF):
                    scaler = MinMaxScaler(feature_range=(0, np.max(self.data.neuron_traces.T)))
                    X_scaled = scaler.fit_transform(self.data.neuron_traces.T)
                    transformed_points = mapping.fit_transform(X_scaled)
                else:
                    transformed_points = mapping.fit_transform(self.data.neuron_traces.T)
            else:
                print('The selected model does not project to a 3 dimensional space.')
                return False
        else:
            print('The selected mapping has no attribute \'fit_transform\'. (SKLEARN models are recommended)')
            return False

        print('Points have coordinate shape: ', transformed_points.shape)
        # self.x, self.y, self.z = transformed_points.T
        self.transformed_points = transformed_points.T
        return True

    def _generate_legend(self,
                         blabs=None,
                         diff=False):
        """
        Generates legend handles from earlier created "self.diff_label_counts" or from labels given as a parameter.

        :param blabs: Labels from which the legend handles should be created
        :type blabs: numpy array

        :param diff: Boolean to say if legend for different predictions should be created

        :return: Legend handles
        """
        # if the legend for the difference plot is requested
        if diff:
            y_labels_diff = {
                label: {wrong: count for wrong, count in self.diff_label_counts[c_idx].items() if count}
                for c_idx, label in enumerate(self.data.states)
            }

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.data.colordict[idx],
                                          markersize=10,
                                          label=r'$\mathbf{' + keyval[0] + '}$' + ' to ' +
                                                "; ".join(
                                                    [r"$\mathbf{" + w_behav + "}$" + f"({amount})"
                                                     for w_behav, amount in keyval[1].items()]
                                                )
                                          if keyval[1] else r'$\mathbf{' + keyval[
                                              0] + '}$' + " predictions were always correct")
                               for idx, keyval in enumerate(y_labels_diff.items())]

            return legend_elements

        # Create custom legend handles
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.data.colordict[idx],
                                      markersize=10,
                                      label=r'$\mathbf{' + lab + '}$' + f' ({list(blabs).count(idx)})')
                           for idx, lab in enumerate(self.data.states)]

        return legend_elements

    def _generate_diff_label_counts(self,
                                    diff_predict):
        """
        Generates the counts of wrong predictions by the model from a numpy array were correct predictions are marked as
        "-1" while wrong ones are correctly labeled.

        :param diff_predict: Array with correct predictions (as "-1") and incorrect predictions (as "0", "1", ...)
        :type diff_predict: numpy array
        """
        # Create dictionary to count different predictions for each label
        self.diff_label_counts = {l: {state: 0 for state in self.data.states} for l in np.unique(self.data.B)}
        for idx, wrong_predict in enumerate(diff_predict):
            pred_label = self.B_pred[idx]
            true_label = self.data.B[idx]
            if wrong_predict > -1:
                self.diff_label_counts[true_label][self.data.states[pred_label]] += 1

    def attachBundDLeNet(self,
                        l_dim=3,
                        epochs=2000,
                        window=15,
                        train=True,
                        use_predictor=True):
        """
        Creates a BundDLeNet and trains it if indicated. The tau-model will be used as a mapping for visualizations and if
        indicated the predictor will be used as a prediction model for visualizations.

        :param l_dim: Size of latent dimension (for visualization should be 3)
        :type l_dim: int

        :param epochs: Number of epochs the neural network should be trained
        :type epochs: int

        :param window: Window size used by BundDLeNet input
        :type window: int

        :param train: If the BundDLeNet should be trained

        :param use_predictor: If the Predictor should be used in future visualizations

        :return: Boolean success indicator
        """
        if self.data.fps is None:
            print('In order to attach the BundDLeNet \'self.data.fps\' has to have a value!')
            return False

        time, newX = preprocess_data(self.data.neuron_traces.T, self.data.fps)
        self.X_, self.B_ = prep_data(newX, self.data.B, win=window)
        self.model = BundDLeNet(latent_dim=l_dim, behaviors=len(self.data.states))
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
            self.bn_tau = True
            self.change_mapping(self.model.tau)
            if use_predictor:
                self.useBundDLePredictor()

        return True

    def plot_loss(self):
        """
        Will plot the loss over epochs as total loss, markov loss (loss for predicted Y-t+1 (=lower) and created Y-t+1
        (=upper)) and behavior loss (loss for predicted B-t+1 (=upper) and true label at t+1).
        """
        if self.loss_array is not None:
            plt.figure()
            for i, label in enumerate(
                    ["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$", "Total loss $\mathcal{L}$"]):
                plt.semilogy(self.loss_array[:, i], label=label)
            plt.legend()
            plt.show()
        else:
            print('No model was trained. No loss saved.')

    def train_model(self,
                    epochs=2000,
                    learning_rate=0.001):
        """
        Trains an attached BundDLeNet using the Adam-optimizer.

        :param epochs: Number of epochs for training
        :param epochs: int

        :param learning_rate: Learning rate used by the Adam-optimizer
        :param learning_rate: float

        :return: Boolean success indicator
        """
        if self.model is None:
            print('No model is attached/loaded.')
            return False
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
        self.bn_tau = True
        return True

    def use_mapping_as_input(self):
        """
        Will use the output from the attached tau-model as an input for a new Database object. This is used if the input
        data of the current object could be too large or for exploratory uses

        :return: A new Database object with the transformed points as neuron traces and labels without the first few
        instances, since the window (needed for nc-mcm-training) is cut from them
        """
        if self.bn_tau:
            names = np.asarray([f'axis{i}' for i in range(self.transformed_points.shape[0])])
            print('X', self.transformed_points.shape)
            print('Y', self.B_.shape)
            Y = self.B_
            print('Y-names', self.data.states.shape)
            print('X-names', names.shape)

        elif self.transformed_points is not None:
            print('WARNING! No BundDLeNet!')
            names = np.asarray([f'axis{i}' for i in range(self.transformed_points.shape[0])])
            Y = self.data.B

        else:
            print('No mapping attached!')
            return False

        return Database(self.transformed_points, Y, names, self.data.states, self.data.fps)

    def _add_quivers3D(self,
                       ax,
                       x,
                       y,
                       z,
                       colors=None,
                       draw=True):
        """
        Adds 3D quivers with appropriate size to an axis using 3D input data
        :param ax: Matplotlib-axis to which the quivers are added
        :param x: x-values
        :param y: y-values
        :param z: z-values
        :param colors: color-values
        :return: Axis with quivers added
        """
        if colors is None:
            colors = self.data.colors[:-1]

        dx = np.diff(x)  # Differences between x coordinates
        dy = np.diff(y)  # Differences between y coordinates
        dz = np.diff(z)  # Differences between z coordinates
        # we do this so each arrowhead has the same size independent of the size of the arrow
        lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # We replace eventual zeros
        zero_indices = np.where(lengths == 0)
        epsilon = 1e-8
        lengths[lengths == 0] = epsilon

        if draw:
            lw = 0.8
        else:
            lw = 0

        mean_length = np.mean(lengths)
        lengths = mean_length / lengths
        for idx in range(len(dx)):
            ax.quiver(x[idx], y[idx], z[idx], dx[idx], dy[idx], dz[idx], color=colors[idx],
                      arrow_length_ratio=lengths[idx], alpha=0.8, linewidths=lw)
        return ax

    def make_movie(self,
                   interval=None,
                   save=False,
                   show_legend=False,
                   grid_off=True,
                   quivers=False,
                   draw=True):
        """
        Makes a movie out of each frame in the imaging data. It uses the tau model or a mapping to map the data to a
        3-dimensional space.

        :param interval: Number of milliseconds between each frame in the movie. Important: For a movie with quivers
        only the saved version will satisfy this, since the creation of each frame takes to long in view-mode.
        :type interval: float

        :param save: Boolean if the gif should be saved

        :param show_legend: Boolean if the legend should be shown in movie

        :param grid_off: Boolean if the grid should be shown

        :param quivers: Boolean if quivers should be used, otherwise it will be a scatterplot

        :return: Boolean success indicator
        """
        if self.transformed_points.shape[0] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        if interval is None:
            if self.data.fps is not None:
                print('The movie well be played in real time.')
                interval = 1000 / self.data.fps
            else:
                interval = 10
        self.interval = interval
        self._create_animation(show_legend=show_legend,
                               grid_off=grid_off,
                               quivers=quivers,
                               draw=True)
        plt.show()
        if save:
            name = str(input('What should the movie be called?'))
            self._create_animation(show_legend=show_legend,
                                   grid_off=grid_off,
                                   quivers=quivers,
                                   draw=draw)
            self.save_gif(name)

        return True

    def _create_animation(self,
                          show_legend=False,
                          grid_off=True,
                          quivers=False,
                          draw=True):

        self.scatter = None
        fig, self.movie_ax, legend_elements = self.plot_mapping(show_legend=show_legend,
                                                                grid_off=grid_off,
                                                                quivers=quivers,
                                                                show=False,
                                                                draw=draw)
        self.animation = anim.FuncAnimation(fig, self._update,
                                            fargs=(grid_off, legend_elements, quivers, draw),
                                            frames=self.transformed_points.shape[1],
                                            interval=self.interval)

    def _update(self,
                frame,
                grid_off,
                legend_elements,
                quivers,
                draw):
        """
        Update function to create a frame in the movie.

        :param frame: Index of the frame
        :type frame: int

        :param grid_off: Boolean if grid is shown

        :param legend_elements: Either legend handles or False if no legend is shown

        :return: The Axis the movie is played on
        """
        if self.scatter is not None:
            self.scatter.remove()
            if not draw:
                if quivers:
                    self.movie_ax = self._add_quivers3D(self.movie_ax, *self.transformed_points[:, frame:frame + 2],
                                                        colors=self.plot_colors[frame:frame + 2])
                else:
                    self.movie_ax.scatter(*self.transformed_points[:, frame],
                                          color=self.plot_colors[frame],
                                          s=1,
                                          alpha=0.5)

        x, y, z = self.transformed_points[:, frame]
        self.scatter = self.movie_ax.scatter(x, y, z, s=20, alpha=0.8, color='red')
        # self.scatter = self.movie_ax.scatter(self.x[frame], self.y[frame], self.z[frame], s=20, alpha=0.8, color='red')
        self.movie_ax.set_title(f'Frame: {frame}\nBehavior: {self.data.states[self.data.B[frame]]}')

        if legend_elements:
            self.movie_ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=[1,0])

        if not grid_off:
            self.movie_ax.set_xlabel('Axes 1')
            self.movie_ax.set_ylabel('Axes 2')
            self.movie_ax.set_zlabel('Axes 3')

        return self.movie_ax

    def save_gif(self,
                 name,
                 bitrate=1800,
                 dpi=144):
        """
        Saves the movie which was created earlier.

        :param name: Name the movie should be saved under in the "movies" directory.
        :type name: str

        :param bitrate: Bits per Second the movie is processed at by the PillowWriter
        :param bitrate: int

        :param dpi: Dots per Inch = resolution of the gif
        :param dpi: int
        """
        if self.animation is None:
            print('No animation created yet.\nTo create one use \'.make_movie()\'.')
        else:
            print('This may take a while...')
            # self.movie_ax.remove()

            path = 'movies/' + name + '.gif'
            gif_writer = anim.PillowWriter(fps=int(1000 / self.interval), metadata=dict(artist='Me'), bitrate=bitrate)
            self.animation.save(path, writer=gif_writer, dpi=dpi)

    def useBundDLePredictor(self):
        """
        Tries to use the prediction model used in plots to the Predictor of the BundDLeNet. This is normally only used
        if the upon BundDLeNet creation the "use_predictor" parameter was set to False.

        :return: Boolean success indicator
        """
        if self.bn_tau:
            Yt1_upper, Yt1_lower, Bt1_upper = self.model.call(self.X_)
            B_pred_new = np.argmax(Bt1_upper, axis=1).astype(int)
            self.pred_model = self.model
            self.B_pred = B_pred_new
            print(f'Accuracy of BundDLeNet: {round(accuracy_score(self.data.B[self.X_.shape[2]:], self.B_pred), 3)}')

            return True
        else:
            print('It seems there is no BundDLeNet attached yet. Use \'Visualizer.attachBundDLeNet()\'!')
            return False

    def make_comparison(self,
                        show_legend=False,
                        quivers=True):
        """
        Creates a comparison plot between the True labels and the prediction model which is added/selected.

        :param show_legend: If the legend should be shown

        :param quivers: If quivers should be used, otherwise a scatterplot is used

        :return: Boolean success indicator
        """
        if self.transformed_points.shape[0] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        if self.pred_model is None:
            if self.data.pred_model is not None:
                self.pred_model = self.data.pred_model
                self.B_pred = self.data.B_pred
            else:
                print('There was no prediction model attached to the Database-object. Either use fit_model on the '
                      'Database object or use useBundDLePredictor on the Visualizer if you attached a BundDLeNet')
                return False

        # Creating differences in lengths needed for correct plotting of the model/mapping
        reform = False
        win_plot_p = len(self.B_pred) - self.transformed_points.shape[1]
        if win_plot_p < 0:
            # This handles the exception when a BundDLeNet predictor is used for prediction but the points are plotted
            # using a dim-reduction that will not reduce the amount of points
            reform = True
            t = abs(win_plot_p)
            self.transformed_points = self.transformed_points[:, t:]
            win_plot_p = 0
        win_plot_t = len(self.data.B) - self.transformed_points.shape[1]
        win_pred = len(self.data.B) - len(self.B_pred)

        # Calculating differences between prediction and true label - generating the coloring for the 3 plots
        diff_mask = self.data.B[win_plot_t:] != self.B_pred[win_plot_p:]
        diff_predicts = np.where(diff_mask, self.data.B[win_plot_t:], -1)
        self.plot_colors = self.data.colors[win_plot_t:]
        self.colors_pred = [self.data.colordict[val] for val in self.B_pred[win_plot_p:]]
        self.colors_diff_pred = [self.data.colordict[val] if val > -1 else (0.85, 0.85, 0.85) for val in
                                 diff_predicts]
        self._generate_diff_label_counts(diff_predicts)

        # print(f'All these colors should be the same shape: {len(self.plot_colors), len(self.colors_pred), len(self.colors_diff_pred)}')

        fig = plt.figure(figsize=(12, 8))

        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        # True Labels
        ax1.grid(False)
        ax1.set_axis_off()
        if quivers:
            ax1 = self._add_quivers3D(ax1, *self.transformed_points, colors=self.plot_colors)
        else:
            ax1.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5, color=self.plot_colors)
        ax1.set_title(f'True Label')
        # Difference
        ax2.grid(False)
        ax2.set_axis_off()
        if quivers:
            ax2 = self._add_quivers3D(ax2, *self.transformed_points, colors=self.colors_diff_pred)
        else:
            ax2.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5, color=self.colors_diff_pred)
        ax2.set_title(f'\nModel: {type(self.pred_model)}\n'
                      f'Mapping: {type(self.mapping)}\n\n'
                      f'Accuracy at {round(accuracy_score(self.data.B[win_pred:], self.B_pred), 3)}\n')
        # Predictions
        ax3.grid(False)
        ax3.set_axis_off()
        if quivers:
            ax3 = self._add_quivers3D(ax3, *self.transformed_points, colors=self.colors_pred)
        else:
            ax3.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5, color=self.colors_pred)
        ax3.set_title(f'Predicted Label')

        # plot the legend if wanted
        if show_legend:
            legend_1 = self._generate_legend(self.data.B)
            ax1.legend(title='True Labels',
                       handles=legend_1,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.),
                       fontsize='small')

            legend_2 = self._generate_legend(None, diff=True)
            ax2.legend(title='Incorrect Predictions',
                       handles=legend_2,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.),
                       fontsize='small')

            legend_3 = self._generate_legend(self.B_pred)
            ax3.legend(title='Predicted Labels',
                       handles=legend_3,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.),
                       fontsize='small')

        fig.suptitle(f'{self.transformed_points.shape[1]} Frames',
                     fontsize='x-large',
                     fontweight='bold')
        plt.show()
        if len(self.B_pred) - len(self.B_pred[win_plot_p:]) > 0:
            print(f'Some points {len(self.B_pred) - len(self.B_pred[win_plot_p:])} used for accuracy calculation of '
                  f'the model are not plotted, since the mapping does not include them.')

        if reform:
            print(f'The prediction has fewer points than the true labels. Therefore {t} points are not plotted and also'
                  f' not used for accuracy calculation of the model')
            self._transform_points(self.mapping)
        return True

    def save_weights(self,
                     path=None):
        """
        Saves the weights of the BundDLeNet to a given path or as "data/generated/BundDLeNet_model_ + self.data.name"

        :param path: Relative path in the NeuronVisualizer directory with the file name attached.
        :type path: str

        :return: Boolean success indicator
        """
        if self.model is not None:
            if path is None:
                self.model.save_weights('data/generated/BundDLeNet_model_' + self.data.name)
                return True
            else:
                self.model.save_weights(path)
                return True
        else:
            print('No Model created yet.')
            return False


class CustomEnsembleModel:
    """
    This ensemble takes a model and creates binary predictors for each label-combination.
    As a prediction for each instance it gives the most abundant prediction from its sub-models.
    """

    def __init__(self,
                 base_model):
        """
        :param base_model: a model from which the binary classifiers will be built (e.g. Logistic Regression). It needs
        to have the method "fit", "predict" and "predict_proba".
        """
        self.base_model = base_model
        self.combinatorics = []
        self.ensemble_models = []

    def fit(self,
            neuron_traces,
            labels):

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

    def predict(self,
                neuron_traces):

        results = np.zeros((neuron_traces.shape[0], len(self.combinatorics))).astype(int)
        for idx, b_model in enumerate(self.ensemble_models):
            results[:, idx] = b_model.predict(neuron_traces)
        return [np.bincount(results[row, :]).argmax() for row in range(results.shape[0])]

    def predict_proba(self,
                      neuron_traces):

        y_prob_map = np.zeros((neuron_traces.shape[0], len(self.combinatorics)))
        for idx, model in enumerate(self.ensemble_models):
            prob = model.predict_proba(neuron_traces)[:, 0]
            y_prob_map[:, idx] = prob
        return y_prob_map
