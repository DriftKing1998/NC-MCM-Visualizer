import ncmcm as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_type = 'Stationary'
reps = 100
N_states = 15
length = 3000
ticks = [1, 5, 10, 15]
vocab = {0: '1st order Markov',
         1: '2nd order Markov',
         2: '3rd order Markov',
         3: 'Random',
         4: 'Ornstein-Uehlenbeck',
         5: 'Discrete Non-Stationary Markov',
         6: 'Pseudo-Cont. Non-Stationary Markov',
         7: 'Random Walk'}

fig, axes = plt.subplots(2, 4, figsize=(14, 8))

res_s = np.zeros((8, 15, reps))
res = pd.read_excel(f"{test_type}Property_{length}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    #x = res[r].T
    #print(x.shape)
    #print(np.asarray(x)[:, :reps].shape)
    #print(res_s[i, :, :].shape)
    res_s[i, :, :] = res[r].T# np.asarray(x)[:, :reps]
    #res_s[i, :, :] = np.asarray(x)[:, :reps]

###
# for n in range(N_states):
#     print(f'Number of States {n + 1}')
#     for i in range(reps):
#         stat_ou = nc.simulate_stationary_ou(M=length, N=n + 1)
#         p, _ = nc.stationary_property_test(stat_ou, simulations=20, chunks_num=5)
#         res_s[4, n, i] = p
###

vocab = {0: ('1st order Markov', '#fe0000'),
         1: ('2nd order Markov', '#feae45'),
         2: ('3rd order Markov', '#fe9999'),
         3: ('Random', '#fe45ae'),
         4: ('Ornstein-Uehlenbeck', '#0000fe'),
         5: ('Discrete Non-Stationary Markov', '#45aefe'),
         6: ('Pseudo-Cont. Non-Stationary Markov', '#9999fe'),
         7: ('Random Walk', '#ae45fe')}

for type in range(np.prod(axes.shape)):
    x = type % axes.shape[1]
    y = int(np.floor(type / axes.shape[1]))
    # Plotting
    axes[y, x].boxplot(res_s[type, :, :].T,
                       patch_artist=True,
                       boxprops=dict(facecolor=vocab[type][1]),
                       medianprops=dict(color='black'),
                       flierprops=dict(marker='x', alpha=0.5))
    axes[y, x].set_title(f'{vocab[type][0]} process',
                         fontsize=10)
    if y == 1:
        axes[y, x].set_xlabel('Number of States/Clusters')
    if x == 0:
        axes[y, x].set_ylabel('P-Values')
    axes[y, x].axhline(0.05)
    axes[y, x].set_ylim([-0.01, 1.01])
    if len(ticks) > 0:
        axes[y, x].set_xticks(ticks)
        axes[y, x].set_xticklabels(ticks)
plt.suptitle(f'{test_type} Property Test')
plt.tight_layout()
plt.show()
#plt.savefig(f"/Users/michaelhofer/Documents/GitHub/kaobook/stationary_5000")
plt.close()
