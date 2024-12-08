import numpy as np


# Sequence Generation #

def simulate_markov_sequence(M, P=None, N=1, order=1):
    """
        Simulates a Markov process of a given order and samples probabilities from a uniform distribution.

        Parameters:
       
        - M: int, required
            Length of the sequence.

        - P: np.ndarray, optional
            Transition matrix (default is None for random generation).

        - N: int, optional
            Number of states (default is 1).

        - order: int, optional
            Order of the Markov process (default is 1).

        Returns:
       
        - z: np.ndarray
            Simulated sequence.

        - P: np.ndarray
            Used transition matrix.
    """
    if P is None:
        # Initialize the transition matrix for higher-order Markov process
        dims = [N] * (order + 1)
        P = np.random.rand(*dims)
        # Normalize transition matrix probabilities
        P /= np.sum(P, axis=-1, keepdims=True)
    else:
        # Assume P has the correct shape for the given order
        N = P.shape[0]
        order = len(P.shape) - 1

    # Initialize the state sequence
    z = np.zeros(M, dtype=int)
    # Randomly initialize the first 'order' states
    for i in range(order):
        z[i] = np.random.randint(N)

    for m in range(order, M):
        # Extract the previous 'order' states
        prev_states = tuple(z[m - order:m])
        # Get the transition probabilities for the current state
        probabilities = P[prev_states]
        # Choose the next state based on the transition probabilities
        z[m] = np.random.choice(np.arange(N), p=probabilities)

    return z, P


def discrete_non_stationary_process(M, N, changes=4):
    """
    Generate a non-stationary Markov process. Changes in the process are equally split within length M.

    Parameters:
    - M: int, required
        Length of the sequence.

    - N: int, required
        Number of states.

    - changes: int, optional
        Number of changes within the process.

    Returns:
    - seq: list
        Generated sequence.
    """
    if changes == 0:
        return simulate_markov_sequence(M=M, N=N, order=1)[0]

    l = int(np.floor(M / (changes + 1)))
    last = M - (changes * l)
    seq = []

    for c in range(changes):
        seq += list(simulate_markov_sequence(M=l, N=N, order=1)[0])
    seq += list(simulate_markov_sequence(M=last, N=N, order=1)[0])

    return seq


def pseudo_cont_non_stationary_process(M, N, changes=4, epsilon=0.02):
    """
    Generate a non-stationary Markov process. Changes in the process are equally split within length M.

    Parameters:
    - M: int, required
        Length of the sequence.

    - N: int, required
        Number of states.

    - changes: int, optional
        Number of changes within the process.

    - epsilon: float, optional
        Small change to be applied to the transition matrix.

    Returns:
    - seq: list
        Generated sequence.
    """
    # Initial random transition matrix
    P = np.random.rand(N, N)
    # Normalize transition matrix probabilities
    P /= np.sum(P, axis=-1, keepdims=True)

    l = int(np.floor(M / (changes + 1)))
    last = M - (changes * l)
    seq = []

    # Generate a single perturbation matrix
    perturbation = np.random.rand(N, N)
    perturbation /= np.sum(perturbation, axis=-1, keepdims=True)
    row_means = np.mean(perturbation, axis=1, keepdims=True)
    perturbation -= row_means

    # This makes that the maximum is exactly epsilon
    max_val = np.max(np.abs(perturbation), axis=1, keepdims=True)
    perturbation /= max_val
    perturbation = perturbation * epsilon

    def adjust_transition_matrix(P, pert):
        """
        Adjusts the transition matrix P by a perturbation matrix (pert) and normalizes the rows.
        Ensures no NaN values and values are clipped between 0 and 1.
        """
        P += pert
        P = np.clip(P, 0, 1)  # Ensure values are within [0, 1]
        P /= np.sum(P, axis=-1, keepdims=True)  # Normalize to ensure the sum of probabilities is 1
        P = np.nan_to_num(P, nan=1.0 / N)  # Replace NaNs after normalization
        P /= np.sum(P, axis=-1, keepdims=True)  # Normalize to ensure the sum of probabilities is 1
        return P

    for c in range(changes):
        # Adjust each row of the transition matrix P by a value epsilon
        P = adjust_transition_matrix(P, perturbation)
        seq += list(simulate_markov_sequence(M=l, N=N, P=P)[0])

    seq += list(simulate_markov_sequence(M=last, N=N, P=P)[0])

    return seq


def simulate_random_sequence(M, N):
    """
        Simulate a random sequence with N states and length M. Equates to a Markov process of 1st order with equal
        transition probabilities.
    """
    random_sequence = np.random.randint(0, N, size=M)
    return random_sequence


def simulate_stationary_ou(M, N, theta=0.5, mu=0.0, sigma=1):
    """
    Simulates a stationary discrete Markov sequence based on the Ornstein-Uhlenbeck process.

    Parameters:
        M (int): Length of the sequence.
        N (int): Number of possible states (quantized levels).
        theta (float): Mean-reversion strength.
        mu (float): Long-term mean.
        sigma (float): Standard deviation of noise.

    Returns:
        np.ndarray: A discrete stationary Markov sequence of length M with N states.
    """
    X = np.zeros(M)
    for t in range(1, M):
        X[t] = theta * (mu - X[t - 1]) + np.random.normal(scale=sigma)
    states = np.linspace(X.min(), X.max(), N, endpoint=False)
    discrete_X = np.digitize(X, states) - 1
    return discrete_X


def simulate_non_stationary_rw(M, N, sigma=1):
    """
    Simulates a non-stationary discrete sequence using a random walk model.

    Parameters:
        M (int): Length of the sequence.
        N (int): Number of possible states (quantized levels).
        sigma (float): Standard deviation of normal distributed noise.


    Returns:
        np.ndarray: A discrete non-stationary sequence of length M with N states.
    """
    X = np.zeros(M)
    for t in range(1, M):
        X[t] = X[t - 1] + np.random.normal(scale=sigma)
    states = np.linspace(X.min(), X.max(), N, endpoint=False)
    discrete_X = np.digitize(X, states) - 1
    return discrete_X


