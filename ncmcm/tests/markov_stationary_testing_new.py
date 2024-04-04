import matplotlib.pyplot as plt

from ncmcm.classes import *
import os
os.chdir('..')
print(os.getcwd())




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

matlab = Loader(worm_num)
data = Database(*matlab.data)
data.exclude_neurons(b_neurons)

# Adding prediction Model & Cluster BPT
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
# data.fit_model(logreg, ensemble=True)

# data.cluster_BPT(nrep=3, max_clusters=8, plot_markov=False)

def test_params_s(axes, reps, N_states):
    print(f'For {N_states} Clusters!')
    result = np.zeros((4, reps))
    # unadj_result = np.zeros((5, parts-1, reps))
    for i in range(reps):
        # worm_seq = data.xc[:, N_states - 1, i].astype(int)
        true_seq = generate_markov_process(M=5000, N=N_states, order=1)
        rand_seq = simulate_random_sequence(M=5000, N=N_states)
        lag2_seq = generate_markov_process(M=5000, N=N_states, order=2)
        not_stat = non_stationary_process(M=5000, N=N_states, changes=10)

        x, adj_x = test_stationarity(true_seq, plot=False, sim_stationary=800)
        y, adj_y = test_stationarity(rand_seq, plot=False, sim_stationary=800)
        z, adj_z = test_stationarity(lag2_seq, plot=False, sim_stationary=800)
        a, adj_a = test_stationarity(not_stat, plot=False, sim_stationary=800)
        # b, adj_b = test_stationarity(worm_seq, plot=False, sim_stationary=800)

        result[0, i] = np.mean(adj_x)
        result[1, i] = np.mean(adj_y)
        result[2, i] = np.mean(adj_z)
        result[3, i] = np.mean(adj_a)
        # result[4, i] = np.mean(adj_b)

    names = {0: ('1st order Markov', 'blue'),
             1: ('Random', 'red'),
             2: ('2nd order Markov', 'orange'),
             3: ('Non stationary Markov', 'green')}#,
             # 4: (f'Worm {worm_num+1}', 'purple')}
    #names = ['markov', 'random', '2nd order markov', 'non-stationary markov', f'worm_{worm_num+1}']
    for idx, val in names.items():
        x = idx % 2
        y = int(np.floor(idx / 2))
        axes[x, y].boxplot(result[idx, :], positions=[N_states], patch_artist=True, boxprops=dict(facecolor=val[1]))
        axes[x, y].set_title(val[0])
        #lower_bound = np.percentile(result[idx, :], 12.5, axis=1)
        #upper_bound = np.percentile(result[idx, :], 87.5, axis=1)
        #axes.fill_between(list(range(parts + 1))[2:], lower_bound, upper_bound, alpha=0.3)
        axes[x, y].set_ylim(-0.05, 1)
        axes[x, y].axhline(0.05, color='black', linestyle='--')

    return axes



reps = 30
n_states = [2, 5, 8, 11, 14, 17, 20, 23]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.text(0.5, 0.05, 'no. States', ha='center')


for n in n_states:
    _ = test_params_s(axes, reps=reps, N_states=n)

for idx in range(4):
    x = idx % 2
    y = int(np.floor(idx / 2))

    axes[x,y].set_xticks(n_states)
    #axes[i].set_ylim(-0.05, 1)
    #axes[i].axhline(0.05, color='black', linestyle='--')

fig.suptitle(f'Mean p-values of Stationary Test for self determined chunk sizes')
plt.show()
