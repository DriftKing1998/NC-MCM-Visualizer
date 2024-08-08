import numpy as np


# Sequence Generation #

def simulate_markovian(M, P=None, N=1, order=1):
    """
        Simulate a higher-order Markovian process.

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


def make_random_adj_matrices(num_matrices=1000, matrix_shape=(10, 10), sparse=False):
    """
        Generate random adjacency matrices.

        Parameters:
        - num_matrices: int, optional
            Number of matrices to generate.

        - matrix_shape: tuple, optional
            Shape of each matrix.

        - sparse: bool, optional
            Can be applied to get more sparse transition matrices.

        Returns:
        - transition_matrices: list
            List of generated matrices (np.ndarray).
    """
    transition_matrices = []

    for _ in range(num_matrices):
        if sparse:
            random_matrix = np.random.dirichlet(np.ones(matrix_shape[0]), size=matrix_shape[0])
        else:
            random_matrix = np.random.rand(*matrix_shape)

        # Normalize rows to ensure they add up to 1
        transition_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
        transition_matrices.append(transition_matrix)

    return transition_matrices


def non_stationary_process(M, N, changes=4):
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
        return simulate_markovian(M=M, N=N, order=1)[0]

    l = int(np.floor(M / (changes + 1)))
    last = M - (changes * l)
    seq = []

    for c in range(changes):
        seq += list(simulate_markovian(M=l, N=N, order=1)[0])
    seq += list(simulate_markovian(M=last, N=N, order=1)[0])

    return seq


def non_stationary_process2(M, N, changes=4, epsilon=0.01):
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
    #print('Original: \n', P)

    l = int(np.floor(M / (changes + 1)))
    last = M - (changes * l)
    seq = []

    # Generate a single perturbation matrix
    perturbation = np.random.rand(N, N)
    perturbation /= np.sum(perturbation, axis=-1, keepdims=True)
    row_means = np.mean(perturbation, axis=1, keepdims=True)
    perturbation -= row_means

    # Step 3: Scale to fit within [-1, 1]
    #max_val = np.max(np.abs(perturbation), axis=1, keepdims=True)
    #perturbation /= max_val

    perturbation = (perturbation) * epsilon
    #print('Perturbation: \n', perturbation)

    def adjust_transition_matrix(P, perturbation):
        """
        Adjusts the transition matrix P by a perturbation matrix and normalizes the rows.
        Ensures no NaN values and values are clipped between 0 and 1.
        """
        P += perturbation
        P = np.clip(P, 0, 1)  # Ensure values are within [0, 1]
        #P = np.nan_to_num(P, nan=0.0)  # Replace NaNs with 0
        P /= np.sum(P, axis=-1, keepdims=True)  # Normalize to ensure the sum of probabilities is 1
        P = np.nan_to_num(P, nan=1.0 / N)  # Replace NaNs after normalization
        P /= np.sum(P, axis=-1, keepdims=True)  # Normalize to ensure the sum of probabilities is 1
        return P

    for c in range(changes):
        # Adjust each row of the transition matrix P by a value epsilon
        P = adjust_transition_matrix(P, perturbation)
        #print('NEW: \n', P)

        seq += list(simulate_markovian(M=l, N=N, P=P)[0])

    seq += list(simulate_markovian(M=last, N=N, P=P)[0])

    return seq


def simulate_random_sequence(M, N):
    """
        Simulate a random sequence with N states and length M.
    """
    random_sequence = np.random.randint(0, N, size=M)
    return random_sequence

