from .CustomEnsembleModel import CustomEnsembleModel
from ..helpers.general_functions import *
from ..helpers.plotting_functions import *
from ..helpers.processing_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, json
from sklearn.decomposition import PCA
from typing import Optional, List, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import cross_val_score
import networkx as nx
from pyvis.network import Network


class Database:

    def __init__(self,
                 neuron_traces: Union[List[List[float]], np.ndarray],
                 behavior: Union[List[int], List[str], np.ndarray],
                 neuron_names: Optional[Union[List[str], np.ndarray]] = None,
                 behavioral_states: Optional[Union[List[str], np.ndarray]] = None,
                 fps: float = None,
                 name: str = 'nc-mcm',
                 colors: int = None):

        """
        Reads in the data preferably as numpy arrays or Python-lists. If the behavior array consists of strings, a list
        for translation is created (states) and the behavior is transformed into integers (index of states). Also
        creates colors for plotting. The object can be seen as a container for all the data which is used to plot later.

        Parameters:
            
            - neuron_traces: np.ndarray or list, required
                Contains neuronal activity over time as float.

            - behavior:  np.ndarray or list, required
                Contains behavior either as strings or ints.

            - neuron_names: numpy.ndarray or list, optional
                Gives names to the neurons in neuron traces ordered by index as strings.

            - behavioral_states: numpy.ndarray or list, optional
                Gives names to the behaviors in behavior.

            - fps: flaot, optional
                Frames per seconds of the imaging.

            - name: str, optional
                Name used in some plots.

            - colors: int, optional
                If given will select one of three color spectra (1, 2 or 3).
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
            self.states = behavioral_states
        # If no Neuron Names are given, they are generated
        if neuron_names is None:
            self.neuron_names = np.asarray(range(self.neuron_traces.shape[0])).astype(str)
        else:
            self.neuron_names = np.asarray(neuron_names)

        # If no State Names are given, they are generated
        if behavioral_states is None:
            print('State-names are created from behavior-labels. Translation is accessed by\'self.states\'.')
            newB, blabs = make_integer_list(self.B)
            self.states = np.asarray(blabs).astype(str)
        else:
            self.states = np.asarray(behavioral_states)

        self.pred_model = None
        self.cv_scores = None
        self.B_pred = None
        self.yp_map = None
        self.p_memoryless = None
        self.p_stationary = None
        self.xc = None

        self.spectrum = colors
        self.colordict = dict(
            zip(np.unique(self.B), generate_equidistant_colors(len(self.states), color=self.spectrum)))
        self.colors = [self.colordict[val] for val in self.B]

        if self.B.shape[0] != self.neuron_traces.shape[1] or self.neuron_names.shape[0] != self.neuron_traces.shape[0]:
            print('Error! Data seems not to fit together.')
            print(f'Frames of behavior {self.B.shape[0]} are not the same length as frames of neuronal data: '
                  f'{self.neuron_traces.shape[1]}')
            print(f'Or Neuron-names {self.neuron_names.shape[0]} are not the same length as neurons were recorded '
                  f'{self.neuron_traces.shape[0]}')

    def set_colors(self, new_colors):
        """
        Allows to change the colors used.

        Parameters:
            
            - new_colors: numpy.ndarray or list, required
                An array of new colors (same size as unique behaviors).
        """

        if len(new_colors) == len(self.colordict):
            self.colordict = dict(zip(np.unique(self.B), new_colors))
            self.colors = [self.colordict[val] for val in self.B]
        else:
            print(
                f'The size of the colors and different labels do not match!\n{len(new_colors)} != {len(self.colordict)}')

    def exclude_neurons(self,
                        exclude_neurons):
        """
        Excludes specified neurons from the database.

        Parameters:
            
            - exclude_neurons: numpy.ndarray or list, required
                List of neuron names to exclude.
        """

        neuron_names = self.neuron_names
        mask = np.zeros_like(self.neuron_names, dtype='bool')
        for exclude_neuron in exclude_neurons:
            mask = np.logical_or(mask, neuron_names == str(exclude_neuron))
        mask = ~mask
        amount = len(mask) - np.count_nonzero(mask)
        self.neuron_traces = self.neuron_traces[mask]
        self.neuron_names = self.neuron_names[mask]

        print(f'{amount} neurons have been removed.')

    def fit_model(self,
                  base_model,
                  prob_map=True,
                  ensemble=True,
                  cv_folds=0):
        """
        Allows to fit a model which is used to predict behaviors from the neuron traces (accuracy is printed). Its
        probabilities are used for an eventual clustering.

        Parameters:
            
            - base_model: object, required
                The model to be fitted (no specific type provided).

            - prob_map: numpy.ndarray or list, optional
                A transform or array indicating if a probability map is created for each frame. This is used in the
                behavioral probability trajectory clustering (.cluster_BPT()).

            - ensemble: bool, optional
                A boolean indicating if the CustomEnsembleModel should be created. It makes a set of models for
                each behavior, to differentiate it from each of the other behaviors.

            - cv_folds: int, optional
                If cross validation should be applied, this signals the number of cross-folds.

        Returns:
            
            - return: bool
                Boolean success indicator
        """
        if not hasattr(base_model, 'fit'):
            print('Model has no method \'fit\'.')
            return False

        if ensemble:
            self.pred_model = CustomEnsembleModel(base_model)
        else:
            self.pred_model = base_model

        if cv_folds and type(cv_folds) is int:
            self.cv_scores = cross_val_score(self.pred_model, self.neuron_traces.T, self.B, cv=cv_folds,
                                             scoring='accuracy')  # 5-fold cross-validation
            print(f'Mean cross-validation results for {cv_folds} folds:\n'
                  f'\tMean: {np.mean(self.cv_scores)}\n'
                  f'The full scores can be accessed by \'self.cv_scores\'')

        self.pred_model.fit(self.neuron_traces.T, self.B)
        self.B_pred = np.asarray(self.pred_model.predict(self.neuron_traces.T))
        print("Accuracy for full training data:", accuracy_score(self.B, self.B_pred))
        if prob_map:
            # get probabilities and weights
            self.yp_map = self.pred_model.predict_proba(self.neuron_traces.T)
        return True

    def _test_clusters(self,
                       nclusters,
                       reps,
                       kmeans_init,
                       clustering,
                       chunks,
                       sim_m,
                       sim_s,
                       stationary,
                       verbose=1):
        # Clustering in probability space
        if clustering == 'kmeans':
            clusters = KMeans(n_clusters=nclusters, n_init=kmeans_init).fit(self.yp_map)
            xctmp = clusters.labels_
        elif clustering == 'spectral':
            clusters = SpectralClustering(n_clusters=nclusters).fit(self.yp_map)
            xctmp = clusters.row_labels_
        else:
            raise ValueError("Invalid value for 'clustering' parameter. "
                             "It should be either 'kmeans' or 'spectral'. ")

        p, _ = markovian(xctmp, sim_memoryless=sim_m)
        self.p_memoryless[nclusters - 1, reps] = p

        if stationary:
            _, p_adj_s = stationarity(xctmp, chunks=chunks, plot=False, sim_stationary=sim_s, verbose=verbose)
            self.p_stationary[nclusters - 1, reps] = p_adj_s

        self.xc[:, nclusters - 1, reps] = xctmp

    def cluster_BPT_single(self,
                           nclusters,
                           nrep=200,
                           sim_m=500,
                           sim_s=500,
                           chunks=None,
                           clustering='kmeans',
                           kmeans_init='auto',
                           stationary=False,
                           verbose=1):
        if self.yp_map is None:
            print(f'You first need to fit a model (eg. Logistic Regression), '
                  f'which will be used to map to behavioral probability trajectories.\n'
                  f'Use \'.fit_model(<your_model>)\' on this instance before.')
            return False
        M = self.yp_map.shape[0]

        if (self.p_memoryless is None or
                self.p_memoryless.shape[0] < nclusters or
                self.p_memoryless.shape[1] < nrep):
            print('Creating new')
            self.p_memoryless = np.zeros((nclusters, nrep))
            self.p_stationary = np.zeros((nclusters, nrep))
            self.xc = np.zeros((M, nclusters, nrep))

        for reps in range(nrep):
            print(f'Testing markovianity for {nclusters} clusters - repetition {reps + 1}')
            self._test_clusters(nclusters, reps, kmeans_init, clustering, chunks, sim_m, sim_s, stationary,
                                verbose=verbose)
        return True

    def cluster_BPT(self,
                    nrep=200,
                    max_clusters=20,
                    sim_m=500,
                    sim_s=500,
                    chunks=None,
                    clustering='kmeans',
                    kmeans_init='auto',
                    plot_markov=True,
                    stationary=False,
                    verbose=1,
                    **kwargs):
        """
        Clusters behavioral probability trajectories if a model has been fitted on the data.

        Parameters:
            
            - nrep: int, optional
                Repetitions of repeated clustering for each number of clusters.

            - max_clusters: int, optional
                Maximal number of clusters. Clustering will be done for 1 to "max_clusters" clusters.

            - sim_m: int, optional
                Number of simulations for the test statistics of each memory-less test.

            - sim_s: int, optional
                Number of simulations for the test statistics of each stationarity test.

            - chunks: int, optional
                Number of chunks created for the stationarity test (Frobenius norm of the chunks is used).

            - clustering: str, optional
                The clustering algorithm to use (e.g., 'kmeans').

            - kmeans_init: str or int, optional
                Value for the "n_init" parameter of KMeans, defining how often K-means is initialized after
                clustering (the best result is picked by the sklearn package).

            - plot_markov: bool, optional
                Whether to plot the result.

            - stationary: bool, optional
                Whether to also perform a stationarity test.

            - verbose: int, optional
                Level of verbosity for logging information.

        Returns:
            
            - return: bool
                Boolean success indicator
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
            if verbose != 0:
                print("Testing markovianity - repetition ", reps + 1)
            for nrclusters in range(max_clusters):
                # print(f'Clusters: {nrclusters}')
                self._test_clusters(nrclusters + 1, reps, kmeans_init, clustering, chunks, sim_m, sim_s, stationary,
                                    verbose=verbose)

        if plot_markov:
            self._plot_markov(stationary, **kwargs)
        return True

    def _plot_markov(self, stationary=False, **kwargs):
        """
        Creates the markovian plot.
        """
        fig, ax = plt.subplots(**kwargs)
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

        ax.set_title(f'P-value for signs against the Markov property in the sequence of cognitive states')
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
                  png_name=None,
                  **kwargs):
        """
        Creates a plot consisting of 4 plots. The first one shows behavioral labels plotted onto the 2 principal
        components of the neuronal data. The second one shows behavioral labels plotted onto behavioral probability
        trajectories. The third one shows cognitive labels plotted onto behavioral probability trajectories. The
        last one shows cognitive labels plotted onto the 2 principal components of the neuronal data.

        Parameters:
            
            - clusters: int, optional
                Number of cognitive states (clusters) to use.

            - nrep: int, optional
                Repetitions of repeated clustering for each number of clusters.

            - sim_m: int, optional
                Number of simulations for the test statistics of each memory-less test.

            - sim_s: int, optional
                Number of simulations for the test statistics of each stationarity test.

            - save: bool, optional
                Whether the plot should be saved.

            - show: bool, optional
                Whether the plot should be shown.

            - png_name: str, optional
                Name of the plot if it should be saved. If not provided, it will be named
                'step_plot_<self.name>.png'.

        Returns:
            
            - return: bool
                Boolean success indicator
        """

        if self.p_memoryless is None or self.p_memoryless.shape[0] < clusters:
            print('There were no BPT-clusterings computed. It will be done now...')
            self.fit_model(LogisticRegression(solver='lbfgs', max_iter=1000), ensemble=True)
            self.cluster_BPT_single(nrep=nrep, nclusters=clusters, sim_m=sim_m, sim_s=sim_s)

        # Neuronal trajectories preprocessing
        f_size = kwargs.pop('figsize', (16, 8))
        fig, ax = plt.subplots(2, 2, figsize=f_size, **kwargs)

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
        colordict_cog = dict(zip(list(range(clusters)), generate_equidistant_colors(clusters, color=self.spectrum)))
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
                plt.savefig(f'{png_name}.png', format='png')
            else:
                plt.savefig(f'{self.name}.png', format='png')
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
                                 adj_matrix=False,
                                 show=True,
                                 save=False,
                                 interactive=False,
                                 physics=None,
                                 weights_hist=False,
                                 bins=15,
                                 clustering_rep=None,
                                 **kwargs):
        """
        Creates a behavioral state diagram using the defined states as a directed graph.

        Parameters:
            
            - cog_stat_num: int, optional
                Number of cognitive states used.

            - threshold: float, optional
                A threshold used to display edges in the graph; edges with smaller values are not plotted.

            - offset: float, optional
                Distance between clusters in the plot.

            - clustering_rep: int, optional
                Specifies which clustering representation should be used.

            - adj_matrix: bool, optional
                Whether to plot the adjacency matrix.

            - show: bool, optional
                Whether to display the matplotlib plot.

            - save: bool, optional
                Whether to save the matplotlib plot.

            - interactive: bool, optional
                Whether to save an interactive HTML plot.

            - physics: str, optional
                A path to a JSON-File with physics for the pyvis-graph.

        Returns:
            
            - return: bool
                Boolean success indicator
        """

        if self.p_memoryless is None or self.p_memoryless.shape[0] < cog_stat_num:
            print('You need to run the behavioral probability trajectory clustering first (\'.cluster_BPT\').')
            return False

        # make the graph
        node_colors = list(self.colordict.values()) * cog_stat_num
        T, cog_beh_states = adj_matrix_ncmcm(self, cog_stat_num=cog_stat_num, clustering_rep=clustering_rep)

        T_edges = T.copy()
        T_edges[np.diag_indices_from(T_edges)] = 0
        if threshold is None:
            threshold = np.max(T_edges) / 10
            print('Calculated threshold is: ', threshold)
        T_edges[T_edges < threshold] = 0

        # Plot transition distribution if wanted
        if weights_hist:
            tmp = T_edges.copy()
            tmp[tmp == 0] = np.nan
            plt.hist(tmp.reshape(-1, 1), bins=bins)
            plt.title(f'Distribution of edges after removing ones with weight below {np.round(threshold, 5)}')
            plt.ylabel('amount of edges')
            plt.xlabel('edge weights before scaling')
            plt.show(block=False)

        # Create the graph
        G_old = nx.DiGraph()
        G_old.add_nodes_from(cog_beh_states)
        T_edges = T_edges / (np.max(T_edges) / 10)
        nx.from_numpy_array(T_edges, create_using=G_old)
        edge_colors = [node_colors[u] for u, v in G_old.edges()]
        node_sizes = (np.diag(T) / np.max(np.diag(T)) * 250) * (np.sqrt(T.shape[0]) / offset)
        mapping = {node: self.map_names(str(cog_beh_states[node])) for node in G_old.nodes()}
        G = nx.relabel_nodes(G_old, mapping)

        # Reposition Nodes according to subgroups
        cog_groups = []
        for c_num in range(cog_stat_num):
            cog_groups.append([n for n in np.unique(G.nodes) if n.split(':')[0] == 'C' + str(c_num + 1)])
        all_pos = []
        for c_node_group in cog_groups:
            all_pos.append(nx.circular_layout(G.subgraph(c_node_group)))
        adjusted_pos = {}
        degrees_list = np.linspace(0, 360, num=cog_stat_num, endpoint=False)
        for idx, current_pos in enumerate(all_pos):
            adjusted_pos = shift_pos_by(current_pos, adjusted_pos, degrees_list[idx], offset)

        if adj_matrix:
            fig, ax = plt.subplots(1, 2, **kwargs)
            ax_a = ax[0]
            ax_g = ax[1]
            im_a = ax_a.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax_a.set_title('Adjacency Matrix Heatmap')
            plt.colorbar(im_a, ax=ax_a)
            ax_a.set_yticks(np.arange(T.shape[0]), G.nodes)
            ax_a.set_xlabel('Nodes')
            ax_a.set_ylabel('Nodes')
        else:
            fig, ax_g = plt.subplots(**kwargs)

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
        else:
            plt.close()

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
            plt.savefig(f'{name}.png', format='png')
            print(f'Plot has been saved under: {os.getcwd()}/{name}.png')
            plt.close()
        # This right here will create the interactive HTML plot
        if interactive:
            net = Network(directed=True, filter_menu=True, select_menu=True, cdn_resources='remote')
            net.from_nx(G)
            for idx, node in enumerate(net.nodes):
                c, b = node['id'].split(':')
                node['cog_state'] = c
                node['behavior'] = b
                c_int = int(c[1:]) - 1
                b_int = np.where(np.asarray(self.states) == b)[0][0]
                n_idx = (len(self.states) * c_int + b_int)
                r, g, b = self.colordict[b_int]
                node['color'] = f'rgb({r * 255},{g * 255},{b * 255})'
                node['size'] = max(np.sqrt(node_sizes[n_idx]), 2)
                new = {name: int(T[n_idx, i] * (len(self.B) - 1)) for i, name in enumerate(G.nodes)}
                node['title'] = ''.join(f'{k}:{v}\n' for k, v in new.items() if v > 0)

            script_dir = os.path.dirname(os.path.abspath(__file__))

            if physics is None:
                with open(os.path.join(script_dir, "..", "data", "default_physic.json"), 'r') as file:
                    physic = json.load(file)
            elif type(physics) is str:
                with open(physics, 'r') as file:
                    physic = json.load(file)
            else:
                print('ERROR! No valid physics script selected.')
                return None

            physic = json.dumps(physic, indent=2)
            net.set_options(physic)

            name = str(input('File name for the html-plot? '))
            net.show(f'{name}.html', notebook=False)
            print(f'Plot has been saved under: {name}.html')
        return True

    def map_names(self,
                  name):
        """
        Used to generate a state-name from a number
        """
        c, b = name.split('-')
        new_name = f'C{c}:{self.states[int(b)]}'
        return new_name

    def plotting_neuronal_behavioral(self,
                                     vmin=0,
                                     vmax=2,
                                     **kwargs):
        """
        Plots neuronal data and behavioral data as a timeseries.

        Parameters:
            
            - vmin: int, optional
                minimal value for neuronal data values

            - vmax: int, optional
                maximal value for neuronal data values
        """
        f_size = kwargs.pop('figsize', (10, 4))
        fig, axs = plt.subplots(2, 1, figsize=f_size, **kwargs)
        self._neurons(ax=axs[0], vmin=vmin, vmax=vmax)
        self._behavior(ax=axs[1])
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def _behavior(self,
                  ax=None,
                  **kwargs):
        """
        Plots behavioral data as a timeseries onto an axis if given one. Otherwise, a figure will be created and shown.
        """
        show = False
        if ax is None:
            show = True
            f_size = kwargs.pop('figsize', (10, 2))
            fig, ax = plt.subplots(figsize=f_size, **kwargs)

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
                 vmax=2,
                 **kwargs):
        """
        Plots neuronal data as a timeseries onto an axis if given one. Otherwise, a figure will be created and shown.

        Parameters:
            
            - ax: matplotlib.axes.Axes, optional
                Axes to plot the neuronal-timeseries on.

            - vmin: int, optional
                minimal value for neuronal data values

            - vmax: int, optional
                maximal value for neuronal data values
        """
        show = False
        if ax is None:
            show = True
            f_size = kwargs.pop('figsize', (10, 2))
            fig, ax = plt.subplots(figsize=f_size, **kwargs)

        im0 = ax.imshow(self.neuron_traces, aspect='auto', vmin=vmin, vmax=vmax, interpolation='None')
        # tell the colorbar to tick at integers
        # plt.colorbar(im0)
        ax.get_figure().colorbar(im0)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("Neurons")
        ax.set_title("Neuronal activation")
        if show:
            plt.show()
