from ncmcm.helpers.sequence_functions import non_stationary_process2
from ncmcm.ncmcm_classes.Database import *
import os
os.chdir('..')
print(os.getcwd())

# FOR EPSIlON CHANGES
# Define parameters
alpha = 0.01  # significance level
markov_orders = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # effect sizes to test
sample_sizes = [500, 1000, 1500, 2000, 5000, 10000]  # range of sample sizes
states = 5  # 5

# Calculate power for each combination of sample size and effect size
power_results = []

for markov_order in markov_orders:
    power_for_effect_size = []
    for n in sample_sizes:
        print(f'For markov order {markov_order} and sequence length {n} we keep:')
        simulated_sequences = [non_stationary_process2(M=n, N=states, changes=5, epsilon=markov_order) for _ in range(100)]
        markovian_results = [stationarity(s, sim_stationary=200, num_states=states)[0] for s in simulated_sequences]
        non_stat_process = [alpha > m_r_p for m_r_p in markovian_results]
        print(sum(non_stat_process), '/', len(non_stat_process))
        power = sum(non_stat_process) / 100  # proportion of rejections
        power_for_effect_size.append(power)
    power_results.append(power_for_effect_size)

power_results = np.array(power_results)

# Plotting the power curve
fig, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=3)

for i, sample_size in enumerate(sample_sizes):

    x = i % 3
    y = int(i/3)
    y_axis = list(power_results[:, i])
    y_axis.reverse()
    ax2 = ax[y, x].twinx()
    ax2.set_ylabel('Power')
    ax[y, x].plot([0,1,2,3,4,5,6,7,8,9,10], y_axis)
    ax[y, x].set_ylim(0,1)
    ax[y, x].set_title(f'Sequence length: {sample_size}')
    ax[y, x].set_xlabel('Effect size')
    ax[y, x].set_ylabel('Type I error probability')
    ax[y, x].grid(True)
    ax[y, x].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax[y, x].set_xticklabels(['epsilon 1',
                              'epsilon 0.9', 'epsilon 0.8', 'epsilon 0.7',
                              'epsilon 0.6', 'epsilon 0.5', 'epsilon 0.4',
                              'epsilon 0.3', 'epsilon 0.2', 'epsilon 0.1', 'Stat. Markov'],
                             rotation=45,
                             ha='right')

fig.suptitle('Power Curve of Custom Statistical Test')
plt.tight_layout()
plt.show()