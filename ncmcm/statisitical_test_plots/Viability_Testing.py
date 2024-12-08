import ncmcm as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def params_m(axes, reps=5, N_states=10, sim=100, seq_len=3000, ticks=None):
    if ticks is None:
        ticks = []
    result = np.zeros((np.prod(axes.shape), N_states, reps))
    for n in range(N_states):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            true_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=1)
            lag2_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=2)
            lag3_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=3)
            rand_seq = nc.simulate_random_sequence(M=seq_len, N=n + 1)
            stat_ou = nc.simulate_stationary_ou(M=seq_len, N=n + 1)
            not_stat = nc.discrete_non_stationary_process(M=seq_len, N=n + 1, changes=10)
            not_stat2 = nc.pseudo_cont_non_stationary_process(M=seq_len, N=n + 1, changes=10, epsilon=0.05)
            not_stat_rw = nc.simulate_non_stationary_rw(M=seq_len, N=n + 1)

            seqs = [true_seq, lag2_seq, lag3_seq, rand_seq, stat_ou, not_stat, not_stat2, not_stat_rw]

            for idx in range(np.prod(axes.shape)):
                p, _ = nc.markov_property_test(seqs[idx], simulations=sim, states=[i for i in range(n + 1)])
                result[idx, n, i] = p

    vocab = {0: ('1st order Markov', 'blue'),
             1: ('2nd order Markov', 'red'),
             2: ('3rd order Markov', 'orange'),
             3: ('Random', 'yellow'),
             4: ('Ornstein-Uehlenbeck', 'salmon'),
             5: ('Discrete Non-Stationary Markov', 'brown'),
             6: ('Pseudo-Cont. Non-Stationary Markov', 'green'),
             7: ('Random Walk', 'cyan')}

    for type in range(np.prod(axes.shape)):
        x = type % axes.shape[1]
        y = int(np.floor(type / axes.shape[1]))
        # Plotting
        axes[y, x].boxplot(result[type, :, :].T, patch_artist=True,
                           boxprops=dict(facecolor=vocab[type][1]))  # Enable patch_artist
        axes[y, x].set_title(f'{vocab[type][0]} process',
                             fontsize=10)
        if y == 1:
            axes[y, x].set_xlabel('Number of States/Clusters')
        if x == 0:
            axes[y, x].set_ylabel('P-Values')
        axes[y, x].axhline(0.05)
        axes[y, x].set_ylim([0, 1])
        if len(ticks) > 0:
            axes[y, x].set_xticks([1, 5, 10, 15, 20])
            axes[y, x].set_xticklabels([1, 5, 10, 15, 20])
    plt.suptitle('Markov Property Test')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"markov_{seq_len}.png")  # Save the plot to a file
    plt.close()
    return result


def params_s(axes, reps=5, N_states=10, sim=50, seq_len=3000, chunks=None, ticks=None):
    if ticks is None:
        ticks = []
    result = np.zeros((np.prod(axes.shape), N_states, reps))
    for n in range(N_states):
        print(f'Number of States {n + 1}')
        for i in range(reps):
            # true_seq, _ = simulate_markovian(M=1000, P=underlying_process)
            true_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=1)
            lag2_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=2)
            lag3_seq, _ = nc.simulate_markov_sequence(M=seq_len, N=n + 1, order=3)
            rand_seq = nc.simulate_random_sequence(M=seq_len, N=n + 1)
            stat_ou = nc.simulate_stationary_ou(M=seq_len, N=n + 1)
            not_stat = nc.discrete_non_stationary_process(M=seq_len, N=n + 1, changes=5)
            not_stat2 = nc.pseudo_cont_non_stationary_process(M=seq_len, N=n + 1, changes=5, epsilon=0.05)
            not_stat_rw = nc.simulate_non_stationary_rw(M=seq_len, N=n + 1)

            seqs = [true_seq, lag2_seq, lag3_seq, rand_seq, stat_ou, not_stat, not_stat2, not_stat_rw]

            for idx in range(np.prod(axes.shape)):
                p, _ = nc.stationary_property_test(seqs[idx], simulations=sim, test_mode='ks', chunks_num=chunks)
                result[idx, n, i] = p

    vocab = {0: ('1st order Markov', 'blue'),
             1: ('2nd order Markov', 'red'),
             2: ('3rd order Markov', 'orange'),
             3: ('Random', 'yellow'),
             4: ('Ornstein-Uehlenbeck', 'salmon'),
             5: ('Discrete Non-Stationary Markov', 'brown'),
             6: ('Pseudo-Cont. Non-Stationary Markov', 'green'),
             7: ('Random Walk', 'cyan')}

    for type in range(np.prod(axes.shape)):
        x = type % axes.shape[1]
        y = int(np.floor(type / axes.shape[1]))
        # Plotting
        axes[y, x].boxplot(result[type, :, :].T, patch_artist=True,
                           boxprops=dict(facecolor=vocab[type][1]))  # Enable patch_artist
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
    plt.suptitle('Stationary Property Test')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"stationary_{seq_len}.png")  # Save the plot to a file
    plt.close()
    return result


reps = 50
N_states = 15
length = 3000
vocab = {0: '1st order Markov',
         1: '2nd order Markov',
         2: '3rd order Markov',
         3: 'Random',
         4: 'Ornstein-Uehlenbeck',
         5: 'Discrete Non-Stationary Markov',
         6: 'Pseudo-Cont. Non-Stationary Markov',
         7: 'Random Walk'}

fig1, axes1 = plt.subplots(2, 4, figsize=(14, 8))
res_s = params_s(axes1, reps=reps, N_states=N_states,
                 seq_len=length, sim=20, chunks=5,
                 ticks=[1, 5, 10, 15])

with pd.ExcelWriter(f"StationaryProperty_{length}.xlsx", engine="openpyxl") as writer:
    for i in range(res_s.shape[0]):
        df = pd.DataFrame(res_s[i].T)
        df.to_excel(writer, sheet_name=f"{vocab[i]}", index=False,
                    header=[f'{i + 1} states' for i in range(res_s.shape[1])])
#
# fig2, axes2 = plt.subplots(2, 4, figsize=(14, 8))
# res_m = params_m(axes2, reps=reps, N_states=N_states,
#                  seq_len=length, sim=50,
#                  ticks=[1, 5, 10, 15])
#
# with pd.ExcelWriter(f"MarkovProperty_{length}.xlsx", engine="openpyxl") as writer:
#     for i in range(res_m.shape[0]):
#         df = pd.DataFrame(res_m[i, :, :].T)
#         df.to_excel(writer, sheet_name=f"{vocab[i]}", index=False,
#                     header=[f'{i + 1} states' for i in range(res_m.shape[1])])
