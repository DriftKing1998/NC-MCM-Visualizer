import matplotlib.pyplot as plt
from scripts.functions import *
import numpy as np
import random
import statsmodels.stats.multitest as smt
from scripts.classes import *

# Assuming 'A' and 'B' are encoded as strings in your 'Y' variable
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL'
]

worm_num = 0

data1 = Database(worm_num, verbose=1)
data3 = Database(worm_num + 2, verbose=1)
data5 = Database(worm_num + 4, verbose=1)
data1.exclude_neurons(b_neurons)
data3.exclude_neurons(b_neurons)
data5.exclude_neurons(b_neurons)

# Adding prediction Model & Cluster BPT
logreg1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
logreg3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
logreg5 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data1.fit_model(logreg1, binary=True)
data3.fit_model(logreg3, binary=True)
data5.fit_model(logreg5, binary=True)

data1.cluster_BPT(nrep=10, max_clusters=25, plot_markov=False)
data3.cluster_BPT(nrep=10, max_clusters=25, plot_markov=False)
data5.cluster_BPT(nrep=10, max_clusters=25, plot_markov=False)


def test_params_mem(axes, reps=3, N_states=10):
    result = np.zeros((7, reps, N_states))
    for i in range(reps):
        print(f'Repetition number {i + 1}')
        for n in range(N_states):
            print(f'States {n + 1}')
            # true_seq, _ = simulate_markovian(M=1000, P=underlying_process)
            lag1_seq = generate_markov_process(M=3000, N=n + 1, order=1)
            lag2_seq = generate_markov_process(M=3000, N=n + 1, order=2)
            lag3_seq = generate_markov_process(M=3000, N=n + 1, order=3)

            true_seq1 = data1.xc[:, n, i]
            true_seq3 = data3.xc[:, n, i]
            true_seq5 = data5.xc[:, n, i]
            not_stat = non_stationary_process(M=3000, N=n + 1, changes=10)

            adj_lag1, _ = markovian(lag1_seq, sim_memoryless=800)
            adj_lag2, _ = markovian(lag2_seq, sim_memoryless=800)
            adj_lag3, _ = markovian(lag3_seq, sim_memoryless=800)
            adj_worm1, _ = markovian(true_seq1, sim_memoryless=800)
            adj_worm3, _ = markovian(true_seq3, sim_memoryless=800)
            adj_worm5, _ = markovian(true_seq5, sim_memoryless=800)
            adj_not_stat, _ = markovian(not_stat, sim_memoryless=800)

            result[0, i, n] = adj_lag1
            result[1, i, n] = adj_lag2
            result[2, i, n] = adj_lag3
            result[3, i, n] = adj_worm1
            result[4, i, n] = adj_worm3
            result[5, i, n] = adj_worm5
            result[6, i, n] = adj_not_stat

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[0, :, :], axis=0), label='markov')
    lower_bound = np.percentile(result[0, :, :], 12.5, axis=0)
    upper_bound = np.percentile(result[0, :, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[1, :, :], axis=0), label='2lag markov')
    lower_bound = np.percentile(result[1, :], 12.5, axis=0)
    upper_bound = np.percentile(result[1, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[2, :, :], axis=0), label='3lag markov')
    lower_bound = np.percentile(result[2, :], 12.5, axis=0)
    upper_bound = np.percentile(result[2, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[3, :, :], axis=0), label='worm1 BPT-clustering')
    lower_bound = np.percentile(result[3, :], 12.5, axis=0)
    upper_bound = np.percentile(result[3, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[4, :, :], axis=0), label='worm3 BPT-clustering')
    lower_bound = np.percentile(result[4, :], 12.5, axis=0)
    upper_bound = np.percentile(result[4, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[5, :, :], axis=0), label='worm5 BPT-clustering')
    lower_bound = np.percentile(result[5, :], 12.5, axis=0)
    upper_bound = np.percentile(result[5, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.plot(list(range(N_states + 1))[1:], np.mean(result[6, :, :], axis=0), label='non-stationary markov')
    lower_bound = np.percentile(result[6, :], 12.5, axis=0)
    upper_bound = np.percentile(result[6, :], 87.5, axis=0)
    axes.fill_between(list(range(N_states + 1))[1:], lower_bound, upper_bound, alpha=0.3)

    axes.axhline(0.05, color='black', linestyle='--')
    for tmp in list(range(N_states + 1))[1:]:
        axes.axvline(tmp, color='black', alpha=0.1)
    axes.legend()
    return axes


reps = 10
n_states = 25

fig, axes = plt.subplots(figsize=(12, 9))
_ = test_params_mem(axes, reps=reps, N_states=n_states)
axes.set_title(f'From 1 to {n_states} Cognitive States')
plt.show()
