from ..bundlenet import *
from ..ncmcm_classes.Visualizer import Visualizer


def create_visualizer(database,
                      mapping=None,
                      l_dim=3,
                      epochs=2000,
                      window=15,
                      use_predictor=True):
    """
    Takes a Database object and either a mapping to visualize the data (e.g.: PCA) or parameters for a BundDLeNet
    (l_dim, epochs, window) which will be used to visualize the data. If a BundDLeNet is created, it will be used to
    predict behaviors in future plots. Otherwise, the model fitted on the Database-object, will be used as a predictor,
    if it exists.

    Parameters:
       
        - database: Database, required
            A database object from which the Visualizer will be generated

        - mapping: object, required
            A mapping (such as PCA) which is used for the projection into three dimension. It needs to have the
            method 'fit_transform'.

        - l_dim: int, optional
            Latent dimension the BundDLeNet maps to (for visualisation: 3D; For further use: XD)

        - epochs: int, optional
            Epochs for the BundDLeNet

        - window: int, optional
            Window-size for BundDLeNet

        - use_predictor: bool, optional
            If the BundDLeNet Predictor should be used as prediction model

        - discrete: bool, optional
            If the BundDLeNet should expect discrete labels

    Returns:

        - return: Visualizer
        Will return the correctly configured Visualizer object or None
    """

    # If a mapping is provided
    if mapping is not None:
        vs = Visualizer(database, mapping)
    # otherwise a BundDLeNet is created
    else:
        if database.fps is None:
            print('Give \'self.fps\' a value (float) first!')
            return None
        time, newX = preprocess_data(database.neuron_traces.T, database.fps)
        X_, B_ = prep_data(newX, database.B, win=window)
        model = BundDLeNet(latent_dim=l_dim, behaviors=len(database.states))
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

        vs = Visualizer(database, model.tau, transform=False)
        vs.X_ = X_
        vs.B_ = B_
        vs.model = model
        vs.loss_array = loss_array
        vs.tau_model = model.tau
        vs.bn_tau = True
        # I need to do this later, since X_ is not defined yet
        vs._transform_points(vs.mapping)
        if use_predictor:
            vs.use_bundle_predictor()

    return vs


def load_bundle_visualizer(database,
                           weights_path,
                           l_dim=3,
                           window=15,
                           use_predictor=True):
    """
    Takes a path to the weights and parameters for a BundDLeNet (l_dim, window) which will be created and the
    weights will be loaded in. One can also choose if the predictor of the BundDLeNet or the model form the
    Database-object will be used (if present).

    Parameters:
       
        - database: Database, required
            A Database-object with data from which to create the Visualizer.

        - weights_path: str, required
            A path to the "directory + name of the weights file"

        - l_dim: int, optional
            Latent dimension the BundDLeNet maps to (for visualisation: 3D; For further use: XD)

        - window: int, optional
            Window-size for BundDLeNet

        - use_predictor: bool, optional
            Boolean if the BundDLeNet Predictor should be used for plots.

    Returns:
        - return: Visualizer
            Will return the correctly configured Visualizer object or None
    """

    time, newX = preprocess_data(database.neuron_traces.T, database.fps)
    X_, B_ = prep_data(newX, database.B, win=window)
    model = BundDLeNet(latent_dim=l_dim, behaviors=len(database.states))
    model.build(input_shape=X_.shape)

    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f'Error {e}! No such file.')
        return None

    vs = Visualizer(database, model.tau, transform=False)
    vs.X_ = X_
    vs.B_ = B_
    vs.model = model
    vs.tau_model = model.tau
    vs.bn_tau = True
    # I need to do this here, since X_ is not defined yet
    vs._transform_points(vs.mapping)
    if use_predictor:
        vs.use_bundle_predictor()

    return vs