import colorsys
import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels.stats.multitest as smt


# General Functions #

def generate_equidistant_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  # Calculate the hue value
        saturation = 1.0  # Set saturation to 1.0 (fully saturated)
        value = 1.0  # Set value to 1.0 (full brightness)
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # Convert HSV to RGB
        colors.append(rgb_color)
    return colors


def shift_pos_by(old_positioning, new_positioning, degree, offset):
    for node, coords in old_positioning.items():
        new_positioning[node] = (coords[0] + offset * np.cos(np.radians(degree)),
                                 coords[1] + offset * np.sin(np.radians(degree)))
    return new_positioning


# Functions Markov #

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
        print('This should not happen!!!\nERROR in markovian-method')
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
            i = zH0[m]  # depth
            j = zH0[m - 1]  # column
            k = zH0[m - 2]  # row
            PH0[k, j, i] += 1

        PH0 = PH0 / (M - 2)
        Pz1z2H0 = np.sum(PH0, axis=2)  # depth summing

        # I am replacing zeros in Pz1z2H0 with epsilon, so we do not encounter RuntimeWarnings
        epsilon = 1e-8
        Pz1z2H0 = np.where(Pz1z2H0 == 0, epsilon, Pz1z2H0)
        Pz1z2H0 = Pz1z2H0 / np.sum(Pz1z2H0)  # here I normalize it so the sum is 1 again

        # print(PH0)
        # print('divide by:')
        # print(np.tile(Pz1z2H0, (N, 1, 1)))

        P2H0 = PH0 / np.tile(Pz1z2H0, (N, 1, 1))
        # print('is')
        # print(P2H0)
        P2H0 = np.nan_to_num(P2H0)

        TH0[kperm] = sum(np.var(P2H0, axis=0).flatten())
        # TH0[kperm] = sum(np.var(P2H0, axis=2).flatten())

    # compute p-value
    T = np.sum(np.var(P2, axis=2), axis=(0, 1))
    p = 1 - np.mean(T >= TH0)
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
        # print(P.sum(axis=1))

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


def empirical_transition_matrix(sequence, num_states):
    states_appeared = np.unique(sequence)
    transition_matrix = np.zeros((num_states, num_states))
    missing_states = np.setdiff1d(np.arange(num_states), states_appeared)

    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        transition_matrix[current_state, next_state] += 1

    # Normalize rows to ensure they sum up to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix /= row_sums

    # Handling states that never appear
    if len(missing_states) > 0:
        prob_missing_state = 1 / num_states
        for state in missing_states:
            transition_matrix[state, :] = prob_missing_state

    return transition_matrix


def make_random_adj_matrices(num_matrices=1000, matrix_shape=(10, 10)):
    transition_matrices = []

    for _ in range(num_matrices):
        # Generate random numbers
        #transition_matrix = np.random.dirichlet(np.ones(matrix_shape[0]), size=matrix_shape[0])

        random_matrix = np.random.rand(*matrix_shape)

        # Normalize rows to ensure they add up to 1
        transition_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
        # print(transition_matrix)
        transition_matrices.append(transition_matrix)

    return transition_matrices
    # Accessing any transition matrix:
    # For example, to access the first transition matrix: transition_matrices[0]


def test_stationarity(sequence, parts=3, test_statistics=1000, plot=False):
    states = np.unique(sequence)
    num_states = len(states)
    transition_dict = {state: [] for state in np.unique(sequence)}
    for i in range(len(sequence) - 1):
        transition = (sequence[i], sequence[i + 1])
        transition_dict[sequence[i]].append(transition)

    # Split each type of transition for each state into parts
    chunks = [[] for _ in range(parts)]
    for state, transitions in transition_dict.items():
        #random.shuffle(transitions)
        state_chunk_length = len(transitions) // parts
        for p in range(parts - 1):
            start = int(state_chunk_length * p)
            end = int(state_chunk_length * (1 + p))
            chunks[p] += transitions[start:end]
        chunks[parts - 1] += transitions[int(state_chunk_length * (parts - 1)):]

    # Making test statistic
    test_stats = []
    test_matrices = make_random_adj_matrices(num_matrices=test_statistics, matrix_shape=(num_states, num_states))
    for idx1, m1 in enumerate(test_matrices):
        for idx2, m2 in enumerate(test_matrices[idx1 + 1:]):
            m_diff = m1 - m2
            frobenius_norm = np.linalg.norm(m_diff, 'fro')
            test_stats.append(frobenius_norm)

    # The 0.05 percentile for significance
    first_percentile = np.percentile(test_stats, 5)

    # calculate the empirical transition matrices from the chunks
    emp_transition_matrices = []
    for c in chunks:
        emp_m = np.zeros((num_states, num_states))
        for t in c:
            emp_m[t[0], t[1]] += 1
        # Normalize rows to ensure they sum up to 1
        row_sums = emp_m.sum(axis=1, keepdims=True)
        emp_m /= row_sums
        emp_transition_matrices.append(emp_m)

    # calculate frobenius norms between the empirical transition matrices
    frobenius_norms = []
    for idx_1, emp_P1 in enumerate(emp_transition_matrices):
        for idx_2, emp_P2 in enumerate(emp_transition_matrices[idx_1 + 1:]):
            #print(f'We compare matrx {idx_1} to matrix {idx_1+idx_2+1}')
            m_test = emp_P1 - emp_P2
            frobenius_empirical = np.linalg.norm(m_test, 'fro')
            frobenius_norms.append(frobenius_empirical)

    # plot all the results
    if plot:
        plt.hist(test_stats, bins='auto', edgecolor='black')  # Adjust the number of bins as needed
        plt.axvline(0, color='orange', label='True Norm')
        for f in frobenius_norms:
            plt.axvline(f, color='green')
        plt.axvline(f, color='green', label='Frobenius Norm between chunks')
        plt.axvline(first_percentile, color='red', label='First 0.05 percentile')

        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Float Values')
        plt.legend()  # Display the legend
        plt.grid(True)
        plt.show()

    # print(f'Frobenius Norms are at : {frobenius_norms}')
    p_values = [1-(np.sum(test_stats >= f) / len(test_stats)) for f in frobenius_norms]
    _, FDR_adjusted_p_values, _, _ = smt.multipletests(p_values, method='fdr_bh')
    mean_unadjusted_p_value = 1-(np.sum(test_stats >= np.mean(frobenius_norms)) / len(test_stats))

    return mean_unadjusted_p_value, np.mean(FDR_adjusted_p_values)


def simulate_random_sequence(M, N):
    random_sequence = np.random.randint(0, N, size=M)
    return random_sequence


def generate_markov_process(M, N, order=2):
    # Randomly initialize transition matrix for the given number of states
    # transition_matrix = np.random.rand(N, N, N)  # For a 3rd order Markov process
    dims = [N] * (order + 1)
    transition_matrix = np.random.rand(*dims)  # For a 3rd order Markov process

    # Normalize transition matrix probabilities
    transition_matrix /= np.sum(transition_matrix, axis=order, keepdims=True)

    # Initialize initial state randomly
    initial_state = np.random.choice(N)

    # Generate a sequence of states for the Markov process
    states = [initial_state] * order
    # print('states and order:', states, order)

    for _ in range(M - 1):
        prev_states = states[-order:] if len(states) >= order else [initial_state] * (order - len(states))

        # Extract probabilities based on previous 'order' states
        probabilities = transition_matrix[tuple(prev_states)]
        # print('trans-matrix: ', transition_matrix)
        # print('previous states: ', prev_states)
        # Choose the next state based on the probabilities
        # print(list(range(N)))
        # print(probabilities)
        # exit()
        new_state = np.random.choice(list(range(N)), p=probabilities)
        states.append(new_state)

    return states


# Data Processing #


def adj_matrix_ncmcm(data, cog_stat_num=3):
    """
    :param data: data from database
    :param cog_stat_num: number of cognitive states in the plot (e.g.: C1, C2, C3 ...)
    :return cog_beh_states: list of all cognitive-behavioral states (coded as: CCBB)
    :return T: adjacency matrix for the cognitive-behavioral states
    """
    best_clustering_idx = np.argmax(data.p_memoryless[cog_stat_num - 1, :])  # according to mr.markov himself
    cog_states = data.xc[:, cog_stat_num - 1, best_clustering_idx].astype(int)

    b = np.unique(data.B)
    c = np.unique(cog_states)  # =cog_stat_num
    T = np.zeros((len(c) * len(b), len(c) * len(b)))

    # This allows for a maximum of 99 different behaviors
    cog_beh_states = [(cs + 1) * 100 + bs for cs in c for bs in b]

    for m in range(len(data.B) - 1):
        cur_sample = m
        next_sample = m + 1
        cur_state = np.where((cog_states[cur_sample] + 1) * 100 + data.B[cur_sample] == cog_beh_states)[0][0]
        next_state = np.where((cog_states[next_sample] + 1) * 100 + data.B[next_sample] == cog_beh_states)[0][0]
        T[next_state, cur_state] += 1

    # normalize T
    T = T / (len(data.B) - 1)
    T = T.T

    return T, cog_beh_states


def make_integer_list(input_list):
    string_to_int = {}
    integer_list = []

    for s in input_list:
        if s not in string_to_int:
            string_to_int[s] = len(string_to_int)
        integer_list.append(string_to_int[s])

    translation_list = list(string_to_int.keys())

    return integer_list, translation_list


def make_windowed_data(X, B, win=15):
    win += 1
    X_win = np.zeros((X.shape[0] - win + 1, win, X.shape[1]))
    for i, _ in enumerate(X_win):
        X_win[i] = X[i:i + win]
    newB = B[win - 1:]

    newX = X_win[:, :-1, :]
    return newX, newB


# Plotting #


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


def non_stationary_process(M, N, changes=4):
    l = int(np.floor(M / changes))
    last = M - ((changes - 1) * l)
    seq = []

    for c in range(changes-1):
        seq += generate_markov_process(M=l, N=N, order=1)
    seq += generate_markov_process(M=last, N=N, order=1)

    return seq


def test_params_s(axes, parts=10, reps=3, N_states=10, sequence=None, plot_markov=True):
    result = np.zeros((3, parts-1, reps))
    unadj_result = np.zeros((3, parts-1, reps))
    for p in range(parts-1):
        print(f'Parts {p + 2}')
        for i in range(reps):
            if sequence is not None:
                true_seq = sequence.astype(int)
            else:
                true_seq = generate_markov_process(M=3000, N=N_states, order=1)
            rand_seq = simulate_random_sequence(M=3000, N=N_states)
            not_stat = non_stationary_process(M=3000, N=N_states, changes=10)

            #print(true_seq)
            #print(rand_seq)

            x, adj_x = test_stationarity(true_seq, parts=p + 2, plot=plot_markov, test_statistics=1000)
            y, adj_y = test_stationarity(rand_seq, parts=p + 2, plot=False, test_statistics=1000)
            a, adj_a = test_stationarity(not_stat, parts=p + 2, plot=False, test_statistics=1000)

            result[0, p, i] = np.mean(adj_x)
            result[1, p, i] = np.mean(adj_y)
            result[2, p, i] = np.mean(adj_a)

            unadj_result[0, p, i] = x
            unadj_result[1, p, i] = y
            unadj_result[2, p, i] = a

    axes.plot(list(range(parts + 1))[2:], np.mean(result[0, :, :], axis=1), label='markov')
    lower_bound = np.percentile(result[0, :, :], 12.5, axis=1)
    upper_bound = np.percentile(result[0, :, :], 87.5, axis=1)
    axes.fill_between(list(range(parts + 1))[2:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(parts + 1))[2:], np.mean(result[1, :, :], axis=1), label='random')
    lower_bound = np.percentile(result[1, :, :], 12.5, axis=1)
    upper_bound = np.percentile(result[1, :, :], 87.5, axis=1)
    axes.fill_between(list(range(parts + 1))[2:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(parts + 1))[2:], np.mean(result[2, :, :], axis=1), label='non-stationary markov')
    lower_bound = np.percentile(result[2, :, :], 12.5, axis=1)
    upper_bound = np.percentile(result[2, :, :], 87.5, axis=1)
    axes.fill_between(list(range(parts + 1))[2:], lower_bound, upper_bound, alpha=0.3)

    axes.axhline(0.05, color='black', linestyle='--')
    for tmp in list(range(parts + 1))[2:]:
        axes.axvline(tmp, color='black', alpha=0.1)
    axes.legend()
    return axes


def test_params_m(axes, reps=3, N_states=10, sim_markov=200):
    result = np.zeros((4, N_states, reps))
    for n in range(N_states):
        print(f'Number of States {n+1}')
        for i in range(reps):
            # true_seq, _ = simulate_markovian(M=1000, P=underlying_process)
            true_seq = generate_markov_process(M=3000, N=n+1, order=1)
            rand_seq = simulate_random_sequence(M=3000, N=n+1)
            lag2_seq = generate_markov_process(M=3000, N=n+1, order=2)
            not_stat = non_stationary_process(M=3000, N=n+1, changes=10)

            p_markov, _ = markovian(true_seq, K=sim_markov)
            p_random, _ = markovian(rand_seq, K=sim_markov)
            p_markov2, _ = markovian(lag2_seq, K=sim_markov)
            p_not_stat, _ = markovian(not_stat, K=sim_markov)

            result[0, n, i] = p_markov
            result[1, n, i] = p_random
            result[2, n, i] = p_markov2
            result[3, n, i] = p_not_stat

    vocab = {0: 'Markov', 1: 'Random', 2:'2nd order Markov', 3: 'Non stationary Markov'}
    for type in range(4):
        x = type % 2
        y = int(np.floor(type/2))
        # Plotting
        axes[y, x].boxplot(result[type, :, :].T)
        axes[y, x].set_title(f'Probability of being a 1st order Markov process for a {vocab[type]} process',
                             fontsize=10)
        axes[y, x].set_xlabel('Number of States/Clusters')
        axes[y, x].set_ylabel('Probability')
        axes[y, x].axhline(0.05)
    plt.tight_layout()
    plt.show()
    return axes
