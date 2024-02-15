from ncmcm.classes import *


def two_step_test(sequence, ax):
    half = int(np.floor(len(sequence) / 2))
    states = len(np.unique(sequence))

    seq_M = sequence[:half]
    seq_M2 = sequence[half:]

    M = np.ones((states, states))
    M2 = np.ones((states, states))
    M3 = np.ones((states, states))
    M4 = np.ones((states, states))

    # Calcualte M
    for i in range(len(seq_M) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        M[current_state][next_state] += 1
    M /= M.sum(axis=1, keepdims=True)  # Normalize

    # Calcualte M2
    for i in range(len(seq_M2) - 2):
        current_state = seq_M2[i]
        next_state = seq_M2[i + 2]
        M2[current_state][next_state] += 1
    M2 /= M2.sum(axis=1, keepdims=True)  # Normalize

    # Calcualte M3
    for i in range(len(seq_M2) - 3):
        current_state = seq_M2[i]
        next_state = seq_M2[i + 3]
        M3[current_state][next_state] += 1
    M3 /= M3.sum(axis=1, keepdims=True)  # Normalize

    # Calcualte M4
    for i in range(len(seq_M2) - 4):
        current_state = seq_M2[i]
        next_state = seq_M2[i + 4]
        M4[current_state][next_state] += 1
    M4 /= M4.sum(axis=1, keepdims=True)  # Normalize

    calculated_M2 = np.dot(M, M)
    calculated_M3 = np.dot(np.dot(M, M), M)
    calculated_M4 = np.dot(np.dot(np.dot(M, M), M), M)

    # Making test statistic
    test_stats = []
    test_matrices = make_random_adj_matrices(num_matrices=500, matrix_shape=(states, states))
    for idx1, m1 in enumerate(test_matrices):
        for idx2, m2 in enumerate(test_matrices[idx1 + 1:]):
            m_diff = m1 - m2
            frobenius_norm = np.linalg.norm(m_diff, 'fro')
            test_stats.append(frobenius_norm)
    # The 0.05 percentile for significance
    first_percentile = np.percentile(test_stats, 5)
    last_percentile = np.percentile(test_stats, 95)
    end_percentile = np.percentile(test_stats, 100)

    # calculate frobenius norms between the empirical transition matrices
    m_diff = M2 - calculated_M2
    frobenius_norm_M2 = np.linalg.norm(m_diff, 'fro')
    m_diff = M3 - calculated_M3
    frobenius_norm_M3 = np.linalg.norm(m_diff, 'fro')
    m_diff = M4 - calculated_M4
    frobenius_norm_M4 = np.linalg.norm(m_diff, 'fro')

    # frobenius_norms = [frobenius_norm_M2, frobenius_norm_M3]

    # plot all the results
    # print(test_stats)
    if states != 1:
        ax.hist(test_stats, bins='auto', edgecolor='black')  # Adjust the number of bins as needed
        ax.axvline(0, color='orange', label='True Norm')
        ax.axvline(frobenius_norm_M2, color='blue', label='Frobenius Norm M2')
        ax.axvline(frobenius_norm_M3, color='cyan', label='Frobenius Norm M3')
        ax.axvline(frobenius_norm_M4, color='green', label='Frobenius Norm M4')
        ax.axvspan(0, first_percentile, color='green', alpha=0.1, label='0-5 percentile')
        ax.axvspan(first_percentile, last_percentile, color='yellow', alpha=0.1, label='5-95 percentile')
        ax.axvspan(last_percentile, max(test_stats + [frobenius_norm_M4, frobenius_norm_M3, frobenius_norm_M2]), color='red', alpha=0.1, label='95-100 percentile')
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_title(f'{states} States')
    else:
        ax.grid(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_axis_off()

    if states == 2:
        '''ax.axvline(0, color='orange', label='True Norm')
        ax.axvline(0, color='blue', label='Frobenius Norm M2')
        ax.axvline(0, color='cyan', label='Frobenius Norm M3')
        ax.axvline(0, color='green', label='Frobenius Norm M4')
        ax.axvspan(0, 0, color='green', alpha=0.1, label='0-5 percentile')
        ax.axvspan(0, 0, color='yellow', alpha=0.1, label='5-95 percentile')
        ax.axvspan(0, 0, color='red', alpha=0.1, label='95-100 percentile')
        '''
        ax.legend(loc='upper left', bbox_to_anchor=(-1.1, 1.3))  # Display the legend
        #ax.grid(False)
        #ax.legend(loc='upper right', bbox_to_anchor=(0.05, 1.7))  # Display the legend
    # plt.show()
    return ax


# true_seq = generate_markov_process(M=3000, N=4, order=1)
# two_step_test(true_seq)


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

data = Database(*Loader(worm_num).data)
data.exclude_neurons(b_neurons)

# Adding prediction Model & Cluster BPT
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg, binary=True)
nrcluster = 16
data.cluster_BPT(nrep=10, max_clusters=nrcluster, plot_markov=False, sim_m=500)

best_clusterings = []
for i in range(nrcluster):
    best_clustering_idx = np.argmax(data.p_memoryless[i, :])  # according to mr.markov himself
    best_clusterings.append(data.xc[:, i, best_clustering_idx].astype(int))
print(np.asarray(best_clusterings).shape)
fig, ax = plt.subplots(4, 4)

for cl in range(nrcluster):
    x = cl % 4
    y = int(np.floor(cl / 4))
    #xctmp = best_clusterings[:, cl]
    xctmp = best_clusterings[cl]

    two_step_test(xctmp, ax[y, x])

plt.show()
# for r in range(5):
#    xctmp = all_xc_3_clusters[:, r].astype(int)
#    p, _ = markovian(xctmp, K=200)
#    _, p_stationary = test_stationarity(xctmp, parts=2, plot=True)
