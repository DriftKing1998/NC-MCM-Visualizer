from ncmcm.ncmcm_classes.Database import *
import os
import pandas as pd
from seaborn import heatmap
os.chdir('..')
print(os.getcwd())

def ordinal(n):
    return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))



#seq = non_stationary_process2(100, 4, changes=8, epsilon=0.1)
#exit()
# FOR EPSIlON CHANGES
# Define parameters
alpha = 0.05  # significance level
effect_sizes = [50, 25, 20, 15, 10, 5]#[5, 10, 15, 20, 25, 50]
sample_sizes = [500, 1000, 1500, 2000, 5000, 10000]
states = 5  # 5

# Calculate power for each combination of sample size and effect size
power_results = []
table = False

if table:
    sims = 50
    seq_len = 2000
    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    for i, markov_order in enumerate(effect_sizes):
        #print(i, markov_order)
        res = []
        non_stationary_sequences = [non_stationary_process(M=seq_len, N=states, changes=markov_order) for _ in range(sims)]
        stationary_sequences = [simulate_markovian(M=seq_len, N=states, order=1)[0] for _ in range(sims)]
        print('Simulated All')
        stationary_sequences_p = [stationarity(s, sim_stationary=300)[0] for s in stationary_sequences]
        print('Tested Falses')
        non_stationary_sequences_p = [stationarity(s, sim_stationary=300)[0] for s in non_stationary_sequences]
        print('Tested Trues')
        correctly_rejected = [alpha >= m_r_p for m_r_p in stationary_sequences_p]
        type_2_error = [alpha < m_r_p for m_r_p in stationary_sequences_p]
        type_1_error = [alpha >= m_r_p for m_r_p in non_stationary_sequences_p]
        correctly_kept = [alpha < m_r_p for m_r_p in non_stationary_sequences_p]

        y = i % 2
        x = int(i / 2)

        cm = pd.DataFrame([[sum(correctly_rejected), sum(type_2_error)],
                           [sum(type_1_error), sum(correctly_kept)]],
                          index=['HO False', 'HO True'],
                          columns=['H0 rejected', 'H0 kept'])
        annotation = np.array([['Correct', 'Type II error'], ['Type I error', 'Correct']])
        data = np.asarray(cm)
        formatted_text = (np.asarray(["{0}\n{1:.2f}%".format(text, (data / (sims*2)) * 100) for text, data in
                                      zip(annotation.flatten(), data.flatten())])).reshape(2, 2)

        heatmap(cm, annot=formatted_text, fmt="", ax=ax[x, y], cbar=False, cmap='Blues', center=sims/2)
        ax[x, y].set_title(
            f'{markov_order} discrete changes')
    plt.suptitle(f'Sequence Length {seq_len} and {sims*2} simulations per plot')
    plt.tight_layout()
    plt.show()

    exit()

power_for_markov = []
for n in sample_sizes:
    print(f'For random process and sequence length {n} we reject:')
    simulated_sequences = [simulate_random_sequence(M=n, N=states) for _ in range(100)]
    markovian_results = [stationarity(s, sim_stationary=200)[0] for s in simulated_sequences]
    non_stat_process = [alpha > m_r_p for m_r_p in markovian_results]
    print(sum(non_stat_process), '/', len(non_stat_process))
    power = sum(non_stat_process) / 100  # proportion of rejections
    power_for_markov.append(power)

power_results.append(power_for_markov)

for effect_size in effect_sizes:
    power_for_effect_size = []
    for n in sample_sizes:
        print(f'For markov order {effect_size} and sequence length {n} we reject:')
        simulated_sequences = [non_stationary_process(M=n, N=states, changes=effect_size) for _ in range(100)]
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
    ax[y, x].plot([0,1,2,3,4,5,6], y_axis)
    ax[y, x].set_ylim(0,1)
    ax[y, x].set_title(f'Sequence length: {sample_size}')
    ax[y, x].set_xlabel('Effect size')
    ax[y, x].set_ylabel('Power')
    ax[y, x].grid(True)
    ax[y, x].set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax[y, x].set_xticklabels(['5 changes', '10 changes', '15 changes',
                              '20 changes', '25 changes', '50 changes',
                              'Random Process'],
                             rotation=45,
                             ha='right')

fig.suptitle('Power Curve of Custom Statistical Test')
plt.tight_layout()
plt.show()