import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import numpy as np
from scipy import signal
from tqdm import tqdm


class BundDLeNet(Model):
    """Behaviour and Dynamical Learning Network (BundDLeNet) model.

    This model represents the BundDLeNet's architecture for deep learning and is based on the commutativity
    diagrams. The resulting model is dynamically consistent (DC) and behaviourally consistent (BC) as per
    the notion described in the paper.

    Args:
        latent_dim (int): Dimension of the latent space.
        behaviors (int): Amount of different behaviors.
    """

    def __init__(self, latent_dim, behaviors):
        super(BundDLeNet, self).__init__()
        self.latent_dim = latent_dim
        # This is the tau mapping from neurons to latent dimension using the windowed input (default 15)
        self.tau = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1),
            layers.GaussianNoise(0.05)
        ])
        # This is the T_Y model that predicts the difference of the current state (t) and the next state (t+1)
        self.T_Y = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1),
        ])
        # This is a model which predicts the behavior from the latent dimension
        self.predictor = tf.keras.Sequential([
            layers.Dense(behaviors, activation='linear')
        ])

    def call(self, X):
        # Upper arm of commutativity diagram
        Yt1_upper = self.tau(X[:, 1]) # uses the second dimension of the preprocessed X
        Bt1_upper = self.predictor(Yt1_upper)

        # Lower arm of commutativity diagram
        Yt_lower = self.tau(X[:, 0]) # uses the first dimension of the preprocessed X
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper


class BundDLeNetTrainer:
    """Trainer for the BundDLeNet model.

    This class handles the training process for the BundDLeNet model.

    Args:
        model: Instance of the BundDLeNet class.
        optimizer: Optimizer for model training.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @tf.function
    def train_step(self, x_train, b_train_1, gamma):
        with tf.GradientTape() as tape:
            yt1_upper, yt1_lower, bt1_upper = self.model(x_train, training=True)
            DCC_loss, behaviour_loss, total_loss = bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return DCC_loss, behaviour_loss, total_loss


### FUNCTIONS ####
def preprocess_data(X, fps):
    """Preprocess the input data by applying bandpass filtering.

    Args:
        X: Input data.
        fps (float): Frames per second.

    Returns:
        numpy.ndarray: Preprocessed data after bandpass filtering.
    """
    time = 1 / fps * np.arange(0, X.shape[0])
    filtered = bandpass(X.T, f_l=1e-10, f_h=0.05, sampling_freq=fps).T

    return time, filtered


def prep_data(X, B, win=15):
    """
    Prepares the data for the BundDLeNet algorithm by formatting the input neuronal and behavioral traces.

    Parameters:
        X : np.ndarray
            Raw neuronal traces of shape (n, t), where n is the number of neurons and t is the number of time steps.
        B : np.ndarray
            Raw behavioral traces of shape (t,), representing the behavioral data corresponding to the neuronal
            traces.
        win : int, optional
            Length of the window to feed as input to the algorithm. If win > 1, a slice of the time series is used
            as input.

    Returns:
        X_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win is the length of each window,
            and n is the number of neurons.
        B_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents
            the behavioral data corresponding to the next time step in the paired neuronal traces.

    """
    win += 1
    X_win = np.zeros((X.shape[0] - win + 1, win, X.shape[1]))
    for i, _ in enumerate(X_win):
        X_win[i] = X[i:i + win]

    Xwin0, Xwin1 = X_win[:, :-1, :], X_win[:, 1:, :]
    B_1 = B[win - 1:]
    X_paired = np.array([Xwin0, Xwin1])
    X_paired = np.transpose(X_paired, axes=(1, 0, 2, 3))

    return X_paired, B_1


def bandpass(traces, f_l, f_h, sampling_freq):
    """
    Apply a bandpass filter to the input traces.

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """
    cut_off_h = f_h * sampling_freq / 2  ## in units of sampling_freq/2
    cut_off_l = f_l * sampling_freq / 2  ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.

    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered


def train_model(X_train, B_train_1, model, optimizer, gamma, n_epochs):
    """
    Training BundDLeNet

    Args:
        X_train: Training input data.
        B_train_1: Training output data.
        model: Instance of the BundDLeNet class.
        optimizer: Optimizer for model training.
        gamma (float): Weight for the DCC loss component.
        n_epochs (int): Number of training epochs.
        # pca_initialisation (bool)
        # best_of_5_init (bool)

    Returns:
        numpy.ndarray: Array of loss values during training.
    """
    # This creates a tuple of features and labels for training
    train_dataset = tf_batch_prep(X_train, B_train_1)

    trainer = BundDLeNetTrainer(model, optimizer)
    loss_array = np.zeros((1, 3))
    epochs = tqdm(np.arange(n_epochs))
    for epoch in epochs:
        for step, (x_train, b_train_1) in enumerate(train_dataset):
            DCC_loss, behaviour_loss, total_loss = trainer.train_step(x_train, b_train_1, gamma=gamma)
            loss_array = np.append(loss_array, [[DCC_loss, behaviour_loss, total_loss]], axis=0)
        epochs.set_description("Losses %f %f %f" % (DCC_loss.numpy(), behaviour_loss.numpy(), total_loss.numpy()))
    loss_array = np.delete(loss_array, 0, axis=0)
    loss_array = loss_array.reshape(n_epochs, int(loss_array.shape[0] // n_epochs), loss_array.shape[-1]).mean(axis=1)
    return loss_array


def tf_batch_prep(X_, B_, batch_size = 100):
    """
    Prepare datasets for TensorFlow by creating batches.

    Parameters:
        X_ : np.ndarray
            Input data of shape (n_samples, ...).
        B_ : np.ndarray
            Target data of shape (n_samples, ...).
        batch_size : int, optional
            Size of the batches to be created. Default is 100.

    Returns:
        batch_dataset : tf.data.Dataset
            TensorFlow dataset containing batches of input data and target data.

    This function prepares datasets for TensorFlow by creating batches. It takes input data 'X_' and target data 'B_'
    and creates a TensorFlow dataset from them.

    The function returns the prepared batch dataset, which will be used for training the TensorFlow model.
    """
    batch_dataset = tf.data.Dataset.from_tensor_slices((X_, B_))
    batch_dataset = batch_dataset.batch(batch_size)
    return batch_dataset


def bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma):
    """Calculate the loss for the BundDLeNet

    Args:
        yt1_upper: Output from the upper arm of the BundDLeNet.
        yt1_lower: Output from the lower arm of the BundDLeNet.
        bt1_upper: Predicted output from the upper arm of the BundDLeNet.
        b_train_1: True output for training.
        gamma (float): Tunable weight for the DCC loss component.

    Returns:
        tuple: A tuple containing the DCC loss, behavior loss, and total loss.
    """
    mse = tf.keras.losses.MeanSquaredError()
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    DCC_loss = mse(yt1_upper, yt1_lower)
    behaviour_loss = scce(b_train_1, bt1_upper)
    total_loss = gamma * DCC_loss + (1 - gamma) * behaviour_loss
    return gamma * DCC_loss, (1 - gamma) * behaviour_loss, total_loss



