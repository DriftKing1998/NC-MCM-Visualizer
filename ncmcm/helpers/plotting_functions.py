import numpy as np
import matplotlib.pyplot as plt
from .markov_functions import stationarity, markovian
from .sequence_functions import simulate_random_sequence, simulate_markovian, non_stationary_process


# Plotting #

def remove_grid(ax):
    ax.grid(False)
    ax.set_axis_off()

def average_markov_plot(markov_array):
    """
        Create a scatter plot of Markov p-values of each worm (input array) with a mean trendline.

        Parameters:
       
        - markov_array: np.ndarray, required
            2D array of Markov p-values.

        Returns:
        None.
    """
    # Scatter plot each row with the index as x-values and the values as y-values
    for i in range(markov_array.shape[0]):
        plt.scatter(np.arange(markov_array.shape[1]), markov_array[i], label=f'Worm {i + 1}')

    mean_trendline = np.mean(markov_array, axis=0)
    plt.plot(np.arange(markov_array.shape[1]), mean_trendline, color='black', linestyle='--', label='Mean Trendline')

    # Add labels and legend
    plt.xlabel('Clusters/States')
    plt.ylabel('Probability')
    plt.axhline(0.05)
    plt.xticks(ticks=np.arange(0, markov_array.shape[1], 1), labels=np.arange(1, markov_array.shape[1] + 1, 1))
    plt.title('Markov Probability for Cognitive States')
    plt.legend()

    # Show the plot
    plt.show()


def parameter_testing_s(axes, parts=10, reps=3, N_states=10, M=3000, sim_s=400, sequence=None, plot_markov=True):
    """
        Test stationary behavior in Markov sequences.

        Parameters:
       
        - axes: matplotlib.axes.Axes, required
            Matplotlib axes.

        - parts: int, optional
            Number of parts to split sequences.

        - reps: int, optional
            Number of test repetitions.

        - N_states: int, optional
            Number of states.

        - M: int, optional
            Size of sequences

        - sim_s: int, optional
            Size of test statistic

        - sequence: np.ndarray, optional
            Input sequence (default is None).

        - plot_markov: bool, optional
            Boolean indicating whether to plot Markov sequences.

        Returns:
       
        - axes: matplotlib.axes.Axes
            Updated Matplotlib axes.
    """
    result = np.zeros((3, parts - 1, reps))
    unadj_result = np.zeros((3, parts - 1, reps))
    for p in range(parts - 1):
        print(f'Parts {p + 2}')
        for i in range(reps):
            if sequence is not None:
                true_seq = sequence.astype(int)
            else:
                true_seq = simulate_markovian(M=M, N=N_states, order=1)
            rand_seq = simulate_random_sequence(M=M, N=N_states)
            not_stat = non_stationary_process(M=M, N=N_states, changes=10)

            x, adj_x = stationarity(true_seq, chunks=p + 2, plot=plot_markov, sim_stationary=sim_s)
            y, adj_y = stationarity(rand_seq, chunks=p + 2, plot=False, sim_stationary=sim_s)
            a, adj_a = stationarity(not_stat, chunks=p + 2, plot=False, sim_stationary=sim_s)

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


def parameter_testing_m(axes, reps=3, N_states=10, sim_markov=200):
    """
        Test memoryless Markov behavior in sequences.

        Parameters:
       
        - axes: matplotlib.axes.Axes, required
            Matplotlib axes.

        - reps: int, optional
            Number of repetitions.

        - N_states: int, optional
            Number of states.

        - sim_markov: int, optional
            Number of simulations for Markov behavior.

        Returns:
       
        - axes: matplotlib.axes.Axes
            Updated Matplotlib axes.
    """
    result = np.zeros((4, N_states, reps))
    for n in range(N_states):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            # true_seq, _ = simulate_markovian(M=1000, P=underlying_process)
            true_seq = simulate_markovian(M=3000, N=n + 1, order=1)
            rand_seq = simulate_random_sequence(M=3000, N=n + 1)
            lag2_seq = simulate_markovian(M=3000, N=n + 1, order=2)
            not_stat = non_stationary_process(M=3000, N=n + 1, changes=10)

            p_markov, _ = markovian(true_seq, sim_memoryless=sim_markov)
            p_random, _ = markovian(rand_seq, sim_memoryless=sim_markov)
            p_markov2, _ = markovian(lag2_seq, sim_memoryless=sim_markov)
            p_not_stat, _ = markovian(not_stat, sim_memoryless=sim_markov)

            result[0, n, i] = p_markov
            result[1, n, i] = p_random
            result[2, n, i] = p_markov2
            result[3, n, i] = p_not_stat

    vocab = {0: 'Markov', 1: 'Random', 2: '2nd order Markov', 3: 'Non stationary Markov'}
    for type in range(4):
        x = type % 2
        y = int(np.floor(type / 2))
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

