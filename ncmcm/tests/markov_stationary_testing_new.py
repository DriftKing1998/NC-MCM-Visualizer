import matplotlib.pyplot as plt

from ncmcm.classes import *
import os
os.chdir('..')
print(os.getcwd())

def test_params_s(axes, reps, N_states, pos):
    print(f'For {N_states} Clusters!')
    result = np.zeros((4, reps))
    for i in range(reps):
        true_seq = generate_markov_process(M=3000, N=N_states, order=1)
        rand_seq = simulate_random_sequence(M=3000, N=N_states)
        lag2_seq = generate_markov_process(M=3000, N=N_states, order=2)
        not_stat = non_stationary_process(M=3000, N=N_states, changes=10)

        x, adj_x = test_stationarity(true_seq, plot=False, sim_stationary=800)
        y, adj_y = test_stationarity(rand_seq, plot=False, sim_stationary=800)
        z, adj_z = test_stationarity(lag2_seq, plot=False, sim_stationary=800)
        a, adj_a = test_stationarity(not_stat, plot=False, sim_stationary=800)

        result[0, i] = np.mean(adj_x)
        result[1, i] = np.mean(adj_y)
        result[2, i] = np.mean(adj_z)
        result[3, i] = np.mean(adj_a)

    names = {0: ('1st order Markov', 'blue'),
             1: ('Random', 'red'),
             2: ('2nd order Markov', 'orange'),
             3: ('Non stationary Markov', 'green')}

    for idx, val in names.items():
        x = idx % 2
        y = int(np.floor(idx / 2))
        axes[x, y].boxplot(result[idx, :], positions=[N_states], patch_artist=True, boxprops=dict(facecolor=val[1]), widths=(2))
        axes[x, y].set_title(val[0])
        axes[x, y].set_ylim(-0.05, 1)
        axes[x, y].axhline(0.05, color='black', linestyle='--')

    return axes



reps = 10
n_states = [2, 5, 8, 11, 14, 17, 20, 23]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.text(0.5, 0.05, 'no. States', ha='center')
fig.text(0.05, 0.5, 'p-values', ha='center', rotation=90)


for i, n in enumerate(n_states):
    _ = test_params_s(axes, reps=reps, N_states=n, pos=i)

for idx in range(4):
    x = idx % 2
    y = int(np.floor(idx / 2))
    axes[x, y].set_xticks(n_states)

fig.suptitle(f'Mean p-values of Stationary Test for self determined chunk sizes')
plt.show()
