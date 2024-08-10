from ..helpers.plotting_functions import *
from ..bundlenet import *
from .Database import Database
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.animation as anim  # FuncAnimation


class Visualizer():

    def __init__(self,
                 data: Database,
                 mapping,
                 transform=True):
        """
        Creates a Visualizer object using a Database object and a mapping.

        Parameters:
            
            - data: Database, required
                Database object with data to be plotted later.

            - mapping: object, required
                Some form of dimensionality reduction, preferably to 3D space, so plotting is possible.

            - transform: bool, optional
                Whether to transform points directly using the mapping. Normally "True" is fine.
        """

        # Setting Attributes
        self.data = data
        self.X_ = None
        self.B_ = None
        # Mapping
        self.mapping = mapping
        if transform:
            self._transform_points(self.mapping)
        else:
            self.transformed_points = None
        # Colors for plotting
        self.window = None
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

        Parameters:
            
            - new_mapping: object, required
                Some sort of dimensionality reduction, preferably to 3D space, so plotting is possible.

        Returns:
            
            - return: bool
                Boolean success indicator
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
                     draw=True,
                     **kwargs):
        """
        Uses the mapping of the Visualizer to plot the datapoints into 3D space.

        Parameters:
            
            - show_legend: bool, optional
                Whether the legend should be shown.

            - grid_off: bool, optional
                Whether the grid should be turned off.

            - quivers: bool, optional
                Whether to use quivers; otherwise, a scatterplot is created.

            - show: bool, optional
                Whether to display the plot. If False, the plot's components will be returned.

        Returns:
            
            - return: bool or tuple
                Either a boolean success indicator or a tuple containing the figure, axis, and legend handles.
        """

        if self.transformed_points.shape[0] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111, projection='3d')
        # We need to trim labs and colors if we have a BundDLe
        if self.transformed_points.shape[1] < len(self.data.colors):
            self.window = len(self.data.colors) - self.transformed_points.shape[1]
        else:
            self.window = 0

        if quivers:
            ax = self._add_quivers3D(ax, *self.transformed_points, colors=self.data.colors[self.window:], draw=draw)
        else:
            ax.scatter(*self.transformed_points, label=self.data.states, color=self.data.colors[self.window:], s=1,
                       alpha=draw)

        # plot the legend if wanted
        if show_legend:
            legend_elements = self._generate_legend(self.data.B)
            ax.legend(handles=legend_elements, fontsize='small', loc='lower center', bbox_to_anchor=[1, 0])
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

        Parameters:
            
            - mapping: Dimensionality reduction mapping for the data points (for visualization mapping to 3D space). Type is either a tf-neural-network or any sklearn-object with method "fit_transform"

        Returns:
            
            - return: bool
                Boolean success indicator
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
        Generates legend handles from earlier created `self.diff_label_counts` or from labels given as a parameter.

        Parameters:
            
            - blabs: numpy.ndarray, optional
                Labels from which the legend handles should be created.

            - diff: bool, optional
                Whether to create a legend for different predictions.

        Returns:
            
            - return: list
                List of legend handles.
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
                                    diff_predict,
                                    window_true_trans,
                                    window_pred_trans):
        """
        Generates the counts of wrong predictions by the model from a numpy array where correct predictions are marked as
        "-1" and incorrect predictions are labeled with other values.

        Parameters:
            
            - diff_predict: numpy.ndarray, required
                Array with correct predictions marked as "-1" and incorrect predictions marked with other values (e.g., "0", "1", ...).

            - window_true_trans: int, required
                Size difference between true labels and the number of transformed points.

            - window_pred_trans: int, required
                Size difference between predicted labels and the number of transformed points.
        """

        # Create dictionary to count different predictions for each label
        self.diff_label_counts = {l: {state: 0 for state in self.data.states} for l in np.unique(self.data.B)}
        for idx, wrong_predict in enumerate(diff_predict):
            pred_label = self.data.B_pred[idx + window_pred_trans]
            true_label = self.data.B[idx + window_true_trans]
            if wrong_predict > -1:
                self.diff_label_counts[true_label][self.data.states[pred_label]] += 1

    def attach_bundle(self,
                      l_dim=3,
                      epochs=2000,
                      window=15,
                      train=True,
                      use_predictor=True):
        """
        Creates a BundDLeNet and trains it if indicated. The tau-model will be used as a mapping for visualizations, and if
        specified, the predictor will be used as a prediction model for visualizations.

        Parameters:
            
            - l_dim: int, optional
                Size of the latent dimension (for visualization, should be 3).

            - epochs: int, optional
                Number of epochs for training the neural network.

            - window: int, optional
                Window size used by BundDLeNet input.

            - train: bool, optional
                Whether the BundDLeNet should be trained.

            - use_predictor: bool, optional
                Whether the predictor should be used in future visualizations.

        Returns:
            
            - return: bool
                Boolean success indicator.
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
                self.use_bundle_predictor()

        return True

    def plot_loss(self,
                  **kwargs):
        """
        Will plot the loss over epochs as total loss, markov loss (loss for predicted Y-t+1 (=lower) and created Y-t+1
        (=upper)) and behavior loss (loss for predicted B-t+1 (=upper) and true label at t+1).
        """
        if self.loss_array is not None:
            plt.figure(**kwargs)
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
        Trains an attached BundDLeNet using the Adam optimizer.

        Parameters:
            
            - epochs: int, optional
                Number of epochs for training.

            - learning_rate: float, optional
                Learning rate used by the Adam optimizer.

        Returns:
            
            - return: bool
                Boolean success indicator.
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

        Parameters:

        Returns:
            
            - return: Database
                A new Database object with the transformed points as neuron traces and labels without the first few
                instances, since the window (needed for nc-mcm-training) is cut from them
        """

        if self.bn_tau:
            names = np.asarray([f'axis{i}' for i in range(self.transformed_points.shape[0])])
            print('New X has shape: ', self.transformed_points.shape)
            print('New Y have shape: ', self.B_.shape)
            Y = self.B_
            print('New Y-names have shape: ', self.data.states.shape)
            print('New X-names have shape: ', names.shape)

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
        Adds 3D quivers with appropriate size to an axis using 3D input data.

        Parameters:
            
            - ax: matplotlib.axes.Axes, required
                Matplotlib axis to which the quivers are added.

            - x: numpy.ndarray or list, required
                x-values.

            - y: numpy.ndarray or list, required
                y-values.

            - z: numpy.ndarray or list, required
                z-values.

            - colors: numpy.ndarray or list, optional
                Color values for the quivers.

            - draw: bool, optional
                Whether to draw the quivers on the axis.

        Returns:
            
            - return: matplotlib.axes.Axes
                Axis with quivers added.
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
                   draw=True,
                   **kwargs):
        """
        Creates a movie from each frame in the imaging data. It uses the tau model or a mapping to project the data into
        a 3-dimensional space.

        Parameters:
            
            - interval: float, optional
                Number of milliseconds between each frame in the movie. Note: For movies with quivers, only the saved
                version will adhere to this interval, as frame creation is too slow in view mode.

            - save: bool, optional
                Whether the GIF should be saved.

            - show_legend: bool, optional
                Whether the legend should be shown in the movie.

            - grid_off: bool, optional
                Whether the grid should be turned off.

            - quivers: bool, optional
                Whether to use quivers; otherwise, a scatterplot will be created.

        Returns:
            
            - return: bool
                Boolean success indicator.
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
                               draw=draw)
        plt.show()
        if save:
            name = str(input('What should the movie be called?'))
            self._create_animation(show_legend=show_legend,
                                   grid_off=grid_off,
                                   quivers=quivers,
                                   draw=draw,
                                   **kwargs)
            self.save_gif(name)

        return True

    def _create_animation(self,
                          show_legend=False,
                          grid_off=True,
                          quivers=False,
                          draw=True,
                          **kwargs):

        self.scatter = None
        fig, self.movie_ax, legend_elements = self.plot_mapping(show_legend=show_legend,
                                                                grid_off=grid_off,
                                                                quivers=quivers,
                                                                show=False,
                                                                draw=draw,
                                                                **kwargs)
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

        Parameters:
            
            - frame: int, required
                Index of the frame.

            - grid_off: bool, optional
                Whether the grid is shown.

            - legend_elements: list or bool, optional
                Either legend handles or `False` if no legend is shown.

            - quivers: bool, optional
                Whether to use quivers in the frame.

            - draw: bool, optional
                Whether to draw the frame.

        Returns:
            
            - return: matplotlib.axes.Axes
                The axis on which the movie is played.
        """

        if self.scatter is not None:
            self.scatter.remove()
            if not draw:
                if quivers:
                    self.movie_ax = self._add_quivers3D(self.movie_ax, *self.transformed_points[:, frame:frame + 2],
                                                        colors=self.data.colors[self.window:][frame:frame + 2])
                else:
                    self.movie_ax.scatter(*self.transformed_points[:, frame],
                                          color=self.data.colors[self.window:][frame],
                                          s=1,
                                          alpha=0.5)

        x, y, z = self.transformed_points[:, frame]
        self.scatter = self.movie_ax.scatter(x, y, z, s=20, alpha=0.8, color='red')
        # self.scatter = self.movie_ax.scatter(self.x[frame], self.y[frame], self.z[frame], s=20, alpha=0.8, color='red')
        self.movie_ax.set_title(f'Frame: {frame}\nBehavior: {self.data.states[self.data.B[frame]]}')

        if legend_elements:
            self.movie_ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=[1, 0])

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
        Saves the movie created earlier.

        Parameters:
            
            - name: str, required
                Name under which the movie should be saved in the "movies" directory.

            - bitrate: int, optional
                Bits per second the movie is processed at by the PillowWriter.

            - dpi: int, optional
                Dots per inch (resolution) of the GIF.
        """

        if self.animation is None:
            print('No animation created yet.\nTo create one use \'.make_movie()\'.')
        else:
            print('This may take a while...')
            # self.movie_ax.remove()
            if name.split('.')[-1] == '.gif':
                path = name
            else:
                path = name + '.gif'
            gif_writer = anim.PillowWriter(fps=int(1000 / self.interval), metadata=dict(artist='Me'), bitrate=bitrate)
            self.animation.save(path, writer=gif_writer, dpi=dpi)

    def use_bundle_predictor(self):
        """
        Tries to use the prediction model used in plots to the Predictor of the BundDLeNet. This is normally only used
        if the upon BundDLeNet creation the "use_predictor" parameter was set to False.

        Parameters:

        Returns:
            
            - return: bool
                Boolean success indicator
        """
        if self.bn_tau:
            Yt1_upper, Yt1_lower, Bt1_upper = self.model.call(self.X_)
            B_pred_new = np.argmax(Bt1_upper, axis=1).astype(int)
            self.data.pred_model = self.model
            self.data.B_pred = B_pred_new
            print(
                f'Accuracy of BundDLeNet: {round(accuracy_score(self.data.B[self.X_.shape[2]:], self.data.B_pred), 3)}')

            return True
        else:
            print('It seems there is no BundDLeNet attached yet. Use \'Visualizer.attachBundDLeNet()\'!')
            return False

    def make_comparison(self,
                        show_legend=True,
                        quivers=True,
                        **kwargs):
        """
        Creates a comparison plot between the true labels and the prediction model that is added or selected. The legend
        created corresponds only to the actually displayed points.

        Parameters:
            
            - show_legend: bool, optional
                Whether the legend should be shown.

            - quivers: bool, optional
                Whether to use quivers; otherwise, a scatterplot is used.

        Returns:
            
            - return: bool
                Boolean success indicator.
        """

        if self.transformed_points.shape[0] != 3:
            print('The mapping does not map into a 3D space.')
            return False

        if self.data.pred_model is None:
            print('There was no prediction model attached to the Database-object. Either use fit_model on the '
                  'Database object or use useBundDLePredictor on the Visualizer if you attached a BundDLeNet')
            return False

        # Creating differences in lengths needed for correct plotting of the model/mapping
        reform = False
        window_pred_trans = len(self.data.B_pred) - self.transformed_points.shape[1]
        if window_pred_trans < 0:
            # This handles the exception when a BundDLeNet predictor is used for prediction but the points are plotted
            # using a dim-reduction that will not reduce the amount of points
            reform = True
            t = abs(window_pred_trans)
            self.transformed_points = self.transformed_points[:, t:]
            window_pred_trans = 0
            print(f'The prediction has fewer points than the true labels. Therefore {t} points are not plotted and also'
                  f' not used for accuracy calculation of the model')
        window_true_trans = len(self.data.B) - self.transformed_points.shape[1]
        window_true_pred = len(self.data.B) - len(self.data.B_pred)

        # Calculating differences between prediction and true label - generating the coloring for the 3 plots
        diff_mask = self.data.B[window_true_trans:] != self.data.B_pred[window_pred_trans:]
        diff_predicts = np.where(diff_mask, self.data.B[window_true_trans:], -1)
        self.colors_pred = [self.data.colordict[val] for val in self.data.B_pred[window_pred_trans:]]
        self.colors_diff_pred = [self.data.colordict[val] if val > -1 else (0.85, 0.85, 0.85) for val in
                                 diff_predicts]
        self._generate_diff_label_counts(diff_predicts, window_true_trans, window_pred_trans)

        fig = plt.figure(**kwargs)
        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        # True Labels
        remove_grid(ax1)
        if quivers:
            ax1 = self._add_quivers3D(ax1, *self.transformed_points, colors=self.data.colors[window_true_trans:])
        else:
            ax1.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5,
                        color=self.data.colors[window_true_trans:])
        ax1.set_title(f'True Label')

        # Difference
        remove_grid(ax2)
        if quivers:
            ax2 = self._add_quivers3D(ax2, *self.transformed_points, colors=self.colors_diff_pred)
        else:
            ax2.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5, color=self.colors_diff_pred)
        ax2.set_title(f'\nModel: {type(self.data.pred_model)}\n'
                      f'Mapping: {type(self.mapping)}\n\n'
                      f'In-sample accuracy at {round(accuracy_score(self.data.B[window_true_pred:], self.data.B_pred), 3)}\n')

        # Predictions
        remove_grid(ax3)
        if quivers:
            ax3 = self._add_quivers3D(ax3, *self.transformed_points, colors=self.colors_pred)
        else:
            ax3.scatter(*self.transformed_points, label=self.data.states, s=1, alpha=0.5, color=self.colors_pred)
        ax3.set_title(f'Predicted Label')

        # plot the legend if wanted
        if show_legend:
            legend_1 = self._generate_legend(self.data.B[window_true_trans:])
            ax1.legend(title='True Labels',
                       handles=legend_1,
                       loc='lower center',
                       bbox_to_anchor=(0, 0.),
                       fontsize='small')

            legend_2 = self._generate_legend(None, diff=True)
            ax2.legend(title='Incorrect Predictions',
                       handles=legend_2,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.),
                       fontsize='small')

            legend_3 = self._generate_legend(self.data.B_pred[window_pred_trans:])
            ax3.legend(title='Predicted Labels',
                       handles=legend_3,
                       loc='lower center',
                       bbox_to_anchor=(1, 0.),
                       fontsize='small')

        fig.suptitle(f'{self.transformed_points.shape[1]} Frames',
                     fontsize='x-large',
                     fontweight='bold')
        plt.show()
        if len(self.data.B_pred) - len(self.data.B_pred[window_pred_trans:]) > 0:
            print(
                f'Some points {len(self.data.B_pred) - len(self.data.B_pred[window_pred_trans:])} used for accuracy calculation of '
                f'the model are not plotted, since the mapping does not include them.')

        if reform:
            self._transform_points(self.mapping)
        return True

    def save_weights(self,
                     path):
        """
        Saves the weights of the BundDLeNet to a given path

        Parameters:
            
            - path: str, required
                Relative path in the NeuronVisualizer directory with the file name attached.

        Returns:
            
            - return: bool
                Boolean success indicator
        """
        if self.model is not None:
            self.model.save_weights(path)
            return True
        else:
            print('No BundDLe-Net created yet.')
            return False

