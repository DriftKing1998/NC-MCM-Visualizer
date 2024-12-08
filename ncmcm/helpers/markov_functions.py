import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from .sequence_functions import simulate_markov_sequence
from .processing_functions import make_integer_list


# Functions Markov #


def markov_property_test(sequence, simulations=200, states=None, return_variances=False):
    """
        Test for 1st order Markovian behavior in a sequence. H0 is that the process is a 1st order markov process.

        Parameters:
            
        - sequence:  np.ndarray, required
            Input sequence.

        - sim_memoryless:  int, optional
            Number of simulations for memoryless Markov behavior test statistic.

        Returns:
            
        - p: float
            Probability of Markovian behavior.

        - P1: np.ndarray
            Transition matrix for first-order Markov behavior.
    """
    sequence = np.asarray(sequence).astype(int)

    Pz0z1z2, states, M, N = compute_transition_matrix_lag2(sequence, states=states)

    # P1 = P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1]) = Pz0z1 / Pz1
    Pz0z1 = np.sum(Pz0z1z2, axis=0)
    Pz1 = np.sum(Pz0z1z2, axis=(0, 2))
    if 0 in Pz1:
        print('This should not happen 1!!!')
    P1 = (Pz0z1 / Pz1.reshape(-1, 1))

    # P2 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2 / Pz1z2
    Pz1z2 = np.sum(Pz0z1z2, axis=2)
    if 0 in Pz1z2:
        print('This should not happen 2!!!')
    P2 = Pz0z1z2 / np.tile(Pz1z2[:, :, np.newaxis], (1, 1, N))

    # Testing
    TH0 = np.zeros(simulations)
    for kperm in range(simulations):
        zH0, _ = simulate_markov_sequence(M=M, P=P1)
        # We need to give the individual states since sometimes not all states are in the simulated sequence
        Pz0z1z2H0, _, _, _ = compute_transition_matrix_lag2(zH0, states=states)
        Pz1z2H0 = np.sum(Pz0z1z2H0, axis=2)
        # P2H0 = P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2]) = Pz0z1z2H0 / Pz1z2H0
        P2H0 = Pz0z1z2H0 / np.tile(Pz1z2H0[:, :, np.newaxis], (1, 1, N))
        TH0[kperm] = np.sum(np.var(P2H0, axis=0).flatten())

    # compute p-value
    T = np.sum(np.var(P2, axis=0).flatten())
    p = 1 - np.mean(T >= TH0)

    if return_variances:
        return p, P1, T, TH0
    else:
        return p, P1


def compute_transition_matrix_lag2(sequence, normalize=True, states=None):
    """
        Compute a transition matrix for a lag-2 Markov process.

        Parameters:
            
        - sequence:  np.ndarray, required
            Input sequence.

        - normalize:  bool, optional
            Boolean to normalize the transition matrix (default is True).

        - states:  int, optional
            List of potential states, needed if a potential state is not present in the input sequence (default ->
            unique values from sequence).

        Returns:
            
        - P: np.ndarray
            Transition matrix.

        - states: np.ndarray
            List of unique states in the sequence.

        - M: int
            Length of the sequence.

        - N: int
            Number of unique states.
    """
    if states is None:
        states = sorted(np.unique(sequence))
    M = len(sequence)
    N = len(states)
    # tensor is created
    P = np.zeros((N, N, N))
    for m in range(2, M):
        i = sequence[m]
        j = sequence[m - 1]
        k = sequence[m - 2]
        # from k to j to i
        P[k, j, i] += 1
    if normalize:
        # This is done here at the start, so it does not need to be checked after each calculation
        epsilon = 1e-8
        P = np.where(P == 0, epsilon, P)
        P = P / np.sum(P)
    return P, states, M, N


def stationary_property_test(sequence, chunks_num=None, plot=False, verbose=0, num_states=None,
                             test_mode='ks', simulations=100):
    """Tests if an input sequence breaks the stationary rule. H0 states that the process could be a stationary process,
    while HA states that the sequence is likely stemming from a non-stationary process.

    The Test divides the original sequence into chunks, where each chunk contains an equal number of state transitions
    per state. For each chunk, a transition matrix is computed. The Frobenius norm is used to quantify the differences
    between these matrices, providing a measure of how much the transition patterns vary across chunks.

    To assess whether the observed differences are statistically significant, the Frobenius norms from the test sequence
    are compared to those from a reference distribution with the same state probabilities as the original sequence. This
    comparison is conducted using a two-sample KS-test and/or a two-sample t-test, evaluating whether the variations in
    transition patterns are larger than expected under stationary behavior.

    A plot can be generated to display the calculations using histograms.

    Note:
        - In testing I found out that with very long sequences (>5000) or with high values of simulations (>1000) the
        test is influenced by the unequal/skewed sample sizes, and becomes very sensitive. As a rule of thumb, try to
        keep the:
            simulations <= 1+1000*np.exp(-0.0007*N)
        where N is the length of the sequence.

        - The t-test only works with chunks_num > 2 and with more than 1 state.


    Args:
        - sequence: Input sequence.
        - chunks_num: Number of parts to split sequence.
        - num_states: If not every state of the state space is present here one can set the state space size (int).
        - test_mode: Specifies what to return:
            - "ks" only returns KS-test results.
            - "ttest" only returns T-test results.
            - "both" returns both KS test and t-test results.
        - simulations: Number of simulations for stationary behavior. Can be set to 'optimal', which will indicate the
        algorithm to calculate the recommended amount of 'simulations'.
        - verbose: Either 0 or 1 and gives additional print-outs for value 1.
        - plot: Boolean indicating whether to plot the results.

    Returns:
        - If `test_mode` is "both":
            (p_value_ks, ks_statistic, p_value_tt, t_statistic)
                - p_value_ks : float
                    P-value for the KS test.
                - ks_statistic : float
                    Effect size (test statistic) for the KS test.
                - p_value_tt : float
                    P-value for the t-test.
                - t_statistic : float
                    Effect size (test statistic) for the t-test.
        - If `test_mode` is "ttest":
            (p_value_tt, t_statistic)
        - If `test_mode` is "ks":
            (p_value_ks, ks_statistic)
    """
    sequence, _ = make_integer_list(sequence)

    if num_states is None:
        states = np.unique(sequence)
        num_states = len(states)

    transition_dict = get_trans_dict(sequence)
    if chunks_num is None:
        chunks_num = calculate_chunks(transition_dict, num_states, verbose=verbose)

    # Split each type of transition for each state into chunks
    chunks = get_chunks(transition_dict, chunks_num)
    #print(f'CHUNKS: {[print(c) for c in chunks]}')

    # calculate the empirical conditional transition matrices from the chunks
    emp_transition_matrices = []
    for c in chunks:
        emp_m = estimate_transition_matrix(c, num_states)
        emp_transition_matrices.append(emp_m)

    frobenius_norms = get_frobenius_norms(emp_transition_matrices)
    #print(f'FROBENIUS NORMS: {frobenius_norms}')

    test_stats = []
    full_emp_matrix = estimate_transition_matrix_from_sequence(sequence)
    #print(f'ESTIMATED MATRIX:\n{full_emp_matrix}')
    if simulations == 'optimal':
        simulations = int(1 + 1000 * np.exp(-0.0007 * len(sequence)))
        if verbose > 0:
            print(f'We calculated an appropriate amount of simulations: {simulations}')
    for _ in range(simulations):
        sim_seq, _ = simulate_markov_sequence(P=full_emp_matrix, M=len(sequence))
        sim_transition_dict = get_trans_dict(sim_seq)
        sim_chunks = get_chunks(sim_transition_dict, chunks_num)
        sim_emp_transition_matrices = []
        for sim_c in sim_chunks:
            sim_emp_m = estimate_transition_matrix(sim_c, num_states)
            sim_emp_transition_matrices.append(sim_emp_m)
        sim_frobenius_norms = get_frobenius_norms(sim_emp_transition_matrices)
        test_stats = test_stats + sim_frobenius_norms

    # plot all the results
    if plot:
        fig, ax = plt.subplots()
        ax.hist(test_stats, bins=20, color='black', density=True, alpha=0.6,
                label='Underlying distribution')
        ax.hist(frobenius_norms, bins=20, color='green', density=True, alpha=0.6,
                label='Frobenius distribution')
        ax.axvline(0, color='orange', label='True Norm')
        ax.axvline(np.mean(test_stats), color='black', label='Mean underlying Frobenius', linestyle='--')
        ax.axvline(np.mean(frobenius_norms), color='green', label='Mean sample Frobenius', linestyle='--')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Float Values')
        ax.legend()
        ax.grid(True)
        plt.show()

    sorted_ref = np.sort(test_stats)
    sorted_sample = np.sort(frobenius_norms)

    if test_mode == "both":
        ks_statistic, p_value_ks = ks_2samp(sorted_ref,
                                            sorted_sample,
                                            alternative='greater')
        t_statistic, p_value_tt = ttest_ind(sorted_sample,
                                            sorted_ref,
                                            alternative='greater',
                                            equal_var=False,
                                            axis=0)
        return p_value_ks, ks_statistic, p_value_tt, t_statistic
    elif test_mode == "ttest":
        t_statistic, p_value_tt = ttest_ind(sorted_sample,
                                            sorted_ref,
                                            alternative='greater',
                                            equal_var=False,
                                            axis=0)
        return p_value_tt, t_statistic
    elif test_mode == "ks":
        ks_statistic, p_value_ks = ks_2samp(sorted_ref,
                                            sorted_sample,
                                            alternative='greater')
        return p_value_ks, ks_statistic
    else:
        print(f'Unknown \'test_mode\' selected, will return KS-Test result.')
        ks_statistic, p_value_ks = ks_2samp(sorted_ref,
                                            sorted_sample,
                                            alternative='greater')
        return p_value_ks, ks_statistic


def get_trans_dict(sequence):
    """
        Generates a dictionary of transitions for each unique state in the input sequence.

        Parameters:

        - sequence:  list or np.ndarray, required
            A sequence of states where transitions between consecutive states are to be recorded.

        Returns:

        - transition_dict:  dict
            A dictionary where each key is a unique state, and its value is a list of transitions from that state.
    """
    transition_dict = {state: [] for state in np.unique(sequence)}
    for i in range(len(sequence) - 1):
        transition = (sequence[i], sequence[i + 1])
        transition_dict[sequence[i]].append(transition)
    return transition_dict


def sim_trans_dict(P, M):
    """
        Generates a dictionary of transitions for a sequence of a certain length (M) from a 1st order conditional
        transition matrix (P).

        Parameters:

        - P:  np.ndarray, required
            A conditional transition matrix from which a sequence of transitions between states is generated.

        - M:  int, required
            Length of the sequence of transitions to be generated.

        Returns:

        - transition_dict:  dict
            A dictionary where each key is a unique state, and its value is a list of transitions from that state.
    """
    states = P.shape[0]
    transition_dict = {state: [] for state in list(range(states))}
    current = np.random.randint(states)
    for m in range(M):
        probabilities = P[current]
        next_state = np.random.choice(np.arange(states), p=probabilities)
        transition = (current, next_state)
        transition_dict[current].append(transition)
        current = next_state
    return transition_dict


def calculate_chunks(transition_dict, num_states, verbose=0):
    """
        Calculates the number of chunks based on the given transition dictionary and number of states.

        Parameters:

        - transition_dict:  dict, required
            A dictionary where each key is a state, and its value is a list of transitions.

        - num_states:  int, required
            The number of unique states in the transition dictionary.

        - verbose:  int, optional (default=0)
            A flag to print the purposed number of chunks if set to 1.

        Returns:

        - purposed_chunks:  int
            The calculated number of chunks based on the transition data and number of states.
    """
    min_length = min(len(lst) for lst in transition_dict.values())
    per_state = min_length / num_states
    purposed_chunks = max(2, int(per_state ** 0.5) + 1)
    if verbose == 1:
        print(f'The algorithm purposes {purposed_chunks} chunks')
    return purposed_chunks


def get_chunks(transition_dict, chunks_num):
    """Splits transitions from each state into a specified number (=chunks_num) of chunks. Chunks are (more or less)
    equally sized lists of transitions from each state. The chunks are split in a way to maintain the sequential order
    of transitions in the individual chunks. If the number of transition from an individual state is not dividable by
    the 'chunks_num', the remainder (=n) will be distributed among the first 'n' chunk(s). As a result, the first chunk
    can be larger than the last chunk by up to the number of unique states.

    Note:
        - The number of chunks should not exceed the number of transitions for the least abundant state. This will very
        likely lead to badly estimated transition probabilities in chunk matrices for some state transitions.

    Example:
        - We want to create 5 chunks. The 'transition_dict' contains 7 transitions from state (1), 6 transitions from
        state (2) and 10 transitions from state (3).

        transition_dict = { 1: [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (1, 3), (1, 2)],
                            2: [(2, 1), (2, 2), (2, 2), (2, 2), (2, 3), (2, 3)],
                            3: [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 1)]}

        The resulting chunks would have:
            - 1 or 2 transitions from state 1;
            - 1 transition from state 2;
            - 2 transitions from state 3;

        chunks = [  [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 3)],
                    [(1, 1), (1, 2), (2, 2), (3, 3), (3, 3)],
                    [(1, 1), (2, 2), (3, 3), (3, 1)],
                    [(1, 3), (2, 3), (3, 3), (3, 1)],
                    [(1, 2), (2, 3), (3, 3), (3, 1)]]

    Args:
        - transition_dict: A dictionary where each key is a state, and its value is a list of transitions from that state.
        - chunks: The number of chunks into which the transitions are to be divided. This value should not exceed the
        minimum number of transitions for any state (=occurrences of any state).

    Returns:
        - chunks: A list where each sublist contains transitions for each chunk.
    """
    chunks = [[] for _ in range(chunks_num)]
    for state, transitions in transition_dict.items():
        state_chunk_length = len(transitions) // chunks_num
        rest = len(transitions) % chunks_num
        chunk_elements = [state_chunk_length + 1] * rest + [state_chunk_length] * (chunks_num - rest)
        start = 0
        for idx, chunk in enumerate(chunks):
            chunk += transitions[start:start + chunk_elements[idx]]
            start += chunk_elements[idx]

    return chunks


def get_frobenius_norms(transition_matrices):
    """
        Computes the Frobenius norms between all pairs of transition matrices within a list.

        Parameters:

        - transition_matrices:  np.ndarray or list of np.ndarray, required
            A list of transition matrices, where each matrix represents transition probabilities between states.

        Returns:

        - frobenius_norms:  list
            A list of Frobenius norms computed for each pair of matrices.
    """
    frobenius_norms = []
    for idx_1, emp_P1 in enumerate(transition_matrices):
        for idx_2, emp_P2 in enumerate(transition_matrices[idx_1 + 1:]):
            m_test = emp_P1 - emp_P2
            frobenius_empirical = np.linalg.norm(m_test, 'fro')
            frobenius_norms.append(frobenius_empirical)
    return frobenius_norms


def estimate_transition_matrix(transitions, num_states):
    """
       Constructs an empirical transition matrix from a list of transitions and normalizes it.

       Parameters:

        - transitions:  list of tuples, required
           A list of transitions, where each transition is a tuple representing a transition between two states.

        - num_states:  int, required
           The total number of unique states in the system.

       Returns:

        - emp_m:  numpy array
           A normalized empirical transition matrix where rows sum to 1. Each entry (i, j) represents the probability of transitioning from state i to state j.
   """
    emp_m = np.zeros((num_states, num_states))
    for t in transitions:
        emp_m[t[0], t[1]] += 1
    if 0 in emp_m:
        emp_m[emp_m == 0] = 1e-8
    row_sums = emp_m.sum(axis=1, keepdims=True)
    emp_m /= row_sums
    return emp_m


def estimate_transition_matrix_from_sequence(sequence):
    """
        Generates an empirical transition matrix directly from a sequence of states and normalizes it.

        Parameters:

        - sequence:  list or numpy array, required
            A sequence of states where transitions between consecutive states are to be used to construct the empirical matrix.

        Returns:

        - emp_m:  numpy array
            A normalized empirical transition matrix where rows sum to 1. Each entry (i, j) represents the probability of transitioning from state i to state j.
    """
    states = np.unique(sequence)
    emp_m = np.zeros((len(states), len(states)))
    for i, s in enumerate(sequence[:-1]):
        s2 = sequence[i + 1]
        emp_m[s, s2] += 1
    if 0 in emp_m:
        emp_m[emp_m == 0] = 1e-8
    row_sums = emp_m.sum(axis=1, keepdims=True)
    emp_m /= row_sums
    return emp_m
