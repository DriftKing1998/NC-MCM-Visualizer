import numpy as np
import matplotlib.pyplot as plt
from .markov_functions import stationary_property_test, markov_property_test
from .sequence_functions import simulate_random_sequence, simulate_markov_sequence, discrete_non_stationary_process, pseudo_cont_non_stationary_process


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


def parameter_testing_m(axes, reps=3, N=10, M=3000, sim_markov=200):
    """
        Test memoryless Markov behavior in sequences.

        Parameters:

        - axes: matplotlib.axes.Axes, required
            Matplotlib axes.

        - reps: int, optional
            Number of repetitions.

        - N_states: int, optional
            Number of states.

        - M: int, optional
            Sequence lengths.

        - sim_markov: int, optional
            Number of simulations for Markov behavior.

        Returns:

        - axes: matplotlib.axes.Axes
            Updated Matplotlib axes.
    """
    result = np.zeros((6, N, reps))
    for n in range(N):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            true_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=1)
            lag2_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=2)
            lag3_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=3)
            rand_seq = simulate_random_sequence(M=M, N=n + 1)
            not_stat = discrete_non_stationary_process(M=M, N=n + 1, changes=10)
            not_stat2 = pseudo_cont_non_stationary_process(M=M, N=n + 1, changes=10, epsilon=0.05)

            p_markov, _ = markov_property_test(true_seq, simulations=sim_markov)
            p_markov2, _ = markov_property_test(lag2_seq, simulations=sim_markov)
            p_markov3, _ = markov_property_test(lag3_seq, simulations=sim_markov)
            p_random, _ = markov_property_test(rand_seq, simulations=sim_markov)
            p_not_stat, _ = markov_property_test(not_stat, simulations=sim_markov)
            p_not_stat2, _ = markov_property_test(not_stat2, simulations=sim_markov)

            result[0, n, i] = p_markov
            result[1, n, i] = p_markov2
            result[2, n, i] = p_markov2
            result[3, n, i] = p_random
            result[4, n, i] = p_not_stat
            result[5, n, i] = p_not_stat2
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 6))
    vocab = {0: 'Markov', 1: '2nd order Markov', 2: '3rd order Markov', 3: 'Random Process',
             4: 'Non stationary (discrete) Markov', 5: 'Non stationary (pseudo-continuous) Markov'}
    for t in range(6):
        x = t % 2
        y = int(np.floor(t / 2))
        # Plotting
        bplot = axes[y, x].boxplot(result[t, :, :].T, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[t])
        axes[y, x].set_title(f'{vocab[t]} process',
                             fontsize=10)
        axes[y, x].set_xlabel('Number of States/Clusters')
        axes[y, x].set_ylabel('P-Values')
        axes[y, x].axhline(0.05)
    plt.tight_layout()
    plt.show()
    return axes


def parameter_testing_s(axes, reps=3, N=10, M=3000, sim_stat=200, changes=15):
    """
        Test memoryless Markov behavior in sequences.

        Parameters:

        - axes: matplotlib.axes.Axes, required
            Matplotlib axes.

        - reps: int, optional
            Number of repetitions.

        - N_states: int, optional
            Number of states.

        - M: int, optional
            Sequence lengths.

        - sim_markov: int, optional
            Number of simulations for Markov behavior.

        Returns:

        - axes: matplotlib.axes.Axes
            Updated Matplotlib axes.
    """
    result = np.zeros((6, N, reps, 2))
    for n in range(N):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            true_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=1)
            lag2_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=2)
            lag3_seq, _ = simulate_markov_sequence(M=M, N=n + 1, order=3)
            rand_seq = simulate_random_sequence(M=M, N=n + 1)
            not_stat = discrete_non_stationary_process(M=M, N=n + 1, changes=changes)
            not_stat2 = pseudo_cont_non_stationary_process(M=M, N=n + 1, changes=changes, epsilon=0.02)

            p_markov, _, p_markov_t, _ = stationary_property_test(true_seq, simulations=sim_stat, test_mode='both')
            p_markov2, _, p_markov2_t, _ = stationary_property_test(lag2_seq, simulations=sim_stat, test_mode='both')
            p_markov3, _, p_markov3_t, _ = stationary_property_test(lag3_seq, simulations=sim_stat, test_mode='both')
            p_random, _, p_random_t, _ = stationary_property_test(rand_seq, simulations=sim_stat, test_mode='both')
            p_not_stat, _, p_not_stat_t, _ = stationary_property_test(not_stat, simulations=sim_stat, test_mode='both')
            p_not_stat2, _, p_not_stat2_t, _ = stationary_property_test(not_stat2, simulations=sim_stat, test_mode='both')

            result[0, n, i, 0] = p_markov
            result[1, n, i, 0] = p_markov2
            result[2, n, i, 0] = p_markov2
            result[3, n, i, 0] = p_random
            result[4, n, i, 0] = p_not_stat
            result[5, n, i, 0] = p_not_stat2

            result[0, n, i, 1] = p_markov_t
            result[1, n, i, 1] = p_markov2_t
            result[2, n, i, 1] = p_markov2_t
            result[3, n, i, 1] = p_random_t
            result[4, n, i, 1] = p_not_stat_t
            result[5, n, i, 1] = p_not_stat2_t

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 6))
    vocab = {0: 'Markov', 1: '2nd order Markov', 2: '3rd order Markov', 3: 'Random Process',
             4: 'Non stationary (discrete) Markov', 5: 'Non stationary (pseudo-continuous) Markov'}
    for t in range(6):
        x = t % 2
        y = int(np.floor(t / 2))
        # Plotting
        positions = np.arange(result.shape[1])
        positions_1 = positions - 0.1
        positions_2 = positions + 0.1
        bplot = axes[y, x].boxplot(result[t, :, :, 0].T, positions=positions_1, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[t])

        bplot = axes[y, x].boxplot(result[t, :, :, 1].T, positions=positions_2, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[t])
        axes[y, x].set_title(f'{vocab[t]} process',
                             fontsize=10)
        axes[y, x].set_xlabel('Number of States/Clusters')
        axes[y, x].set_ylabel('P-Values')
        axes[y, x].axhline(0.05)
        axes[y, x].set_xticks(positions)
        axes[y, x].set_xticklabels(positions + 1)
    plt.tight_layout()
    plt.show()
    return axes
