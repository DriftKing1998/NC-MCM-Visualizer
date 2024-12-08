import ncmcm as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def chunks_s(reps, N_states, length=3000, sims=50, chunks=5):
    print(f'{N_states} STATES')
    result = np.zeros((6, reps, chunks - 1))
    for i in range(reps):
        print(f'{i + 1} Repetition')
        true_seq, _ = nc.simulate_markov_sequence(M=length, N=N_states, order=1)
        lag2_seq, _ = nc.simulate_markov_sequence(M=length, N=N_states, order=2)
        #lag3_seq, _ = nc.simulate_markov_sequence(M=length, N=N_states, order=3)
        #rand_seq = nc.simulate_random_sequence(M=length, N=N_states)
        stat_ou = nc.simulate_stationary_ou(M=length, N=N_states)
        not_stat = nc.discrete_non_stationary_process(M=length, N=N_states, changes=10)
        not_stat2 = nc.pseudo_cont_non_stationary_process(M=length, N=N_states, changes=10, epsilon=0.05)
        not_stat_rw = nc.simulate_non_stationary_rw(M=length, N=N_states)

        seqs = [true_seq, lag2_seq, stat_ou, not_stat, not_stat2, not_stat_rw]

        for idx, seq in enumerate(seqs):
            for c in range(chunks - 1):
                p, _ = nc.stationary_property_test(seq, simulations=sims, chunks_num=c + 2, test_mode='ks')
                result[idx, i, c] = p
    return result


l = 3000
sims = 50
reps = 20
max_chunk = 15
max_states = 10  # -2
chunks_ticks = [n + 2 for n in range(max_chunk - 1)]
vocab = {0: ('1st order Markov', 'blue'),
         1: ('2nd order Markov', 'brown'),
         #2: ('3rd order Markov', 'orange'),
         #3: ('Random', 'yellow'),
         2: ('Ornstein-Uehlenbeck', 'green'),
         3: ('Discrete Non-Stationary Markov', 'orange'),
         4: ('Pseudo-Cont. Non-Stationary Markov', 'salmon'),
         5: ('Random Walk', 'red')}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row_col, N in enumerate(range(2, max_states, 2)):
    N = N + 24
    print(N)
    x = row_col % 2
    y = int(np.floor(row_col / 2))

    result = chunks_s(reps, N, chunks=max_chunk, sims=sims, length=l)

    mean_values = np.mean(result, axis=1)
    ci_25 = np.percentile(result, 25, axis=1)
    ci_75 = np.percentile(result, 75, axis=1)
    print(f'SHAPE MEANS {mean_values.shape}')
    for i in range(result.shape[0]):  # Iterate over axis 0
        color = vocab[i][1]  # Get the corresponding color from vocab
        axes[x, y].plot(chunks_ticks, mean_values[i], label=f"{vocab[i][0]}", color=color)  # Line for the mean
        axes[x, y].fill_between(
            chunks_ticks,  # X-axis values
            ci_25[i], ci_75[i],  # Lower and upper bounds for CI
            color=color, alpha=0.2  # Shaded band
        )
    if x == 1 and y == 1:
        axes[x, y].legend()
    axes[x, y].set_xticks(chunks_ticks)
    axes[x, y].set_xticklabels(chunks_ticks)
    if x == 1:
        axes[x, y].set_xlabel('Amount of Chunks')
    if y == 0:
        axes[x, y].set_ylabel('P-Values')
    axes[x, y].set_title(f'States {N}')
    axes[x, y].legend()

    with pd.ExcelWriter(f"ChunkViability_{N}_states_2.xlsx", engine="openpyxl") as writer:
        for i in range(result.shape[0]):
            df = pd.DataFrame(result[i, :, :])  # Convert the 2D slice to a DataFrame
            df.to_excel(writer, sheet_name=f"{vocab[i][0]}", index=False,
                        header=[f'{i + 2} Chunks' for i in range(result.shape[2])])

#plt.savefig(f"Chunk_Viability_3_{reps}.png")  # Save the plot to a file
plt.close()
#plt.show()
