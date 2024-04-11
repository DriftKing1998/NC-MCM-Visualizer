from ncmcm.functions import *
import numpy as np
import matplotlib.pyplot as plt


def test_params_m(axes, reps=3, N_states=10, sim_markov=200):
    result = np.zeros((4, N_states, reps))
    for n in range(N_states):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            # true_seq, _ = simulate_markovian(M=1000, P=underlying_process)
            true_seq = generate_markov_process(M=10000, N=n + 1, order=1)
            rand_seq = simulate_random_sequence(M=10000, N=n + 1)
            lag2_seq = generate_markov_process(M=10000, N=n + 1, order=2)
            not_stat = non_stationary_process(M=10000, N=n + 1, changes=10)

            p_markov, _ = markovian(true_seq, sim_memoryless=sim_markov)
            p_random, _ = markovian(rand_seq, sim_memoryless=sim_markov)
            p_markov2, _ = markovian(lag2_seq, sim_memoryless=sim_markov)
            p_not_stat, _ = markovian(not_stat, sim_memoryless=sim_markov)

            result[0, n, i] = p_markov
            result[1, n, i] = p_random
            result[2, n, i] = p_markov2
            result[3, n, i] = p_not_stat

    vocab = {0: ('1st order Markov', 'blue'),
             1: ('Random', 'red'),
             2: ('2nd order Markov', 'orange'),
             3: ('non stationary Markov', 'green')}
    for type in range(4):
        x = type % 2
        y = int(np.floor(type / 2))
        # Plotting
        axes[y, x].boxplot(result[type, :, :].T, patch_artist=True, boxprops=dict(facecolor=vocab[type][1]))  # Enable patch_artist
        axes[y, x].set_title(f'Results from the Markov Property Test for a {vocab[type][0]} process',
                             fontsize=10)
        axes[y, x].set_xlabel('Number of States/Clusters')
        axes[y, x].set_ylabel('Probability')
        axes[y, x].axhline(0.05)
    plt.tight_layout()
    plt.show()
    return axes


reps = 20
N_states = 30

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
_ = test_params_m(axes, reps=reps, N_states=N_states)
