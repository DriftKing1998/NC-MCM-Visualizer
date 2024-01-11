from scripts.classes import *

def two_step_test(sequence, ax):
    half = int(np.floor(len(sequence)/2))
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

    if states == 15:
        print(calculated_M2)
        print(calculated_M3)
        print(calculated_M4)

    # Making test statistic
    test_stats = []
    test_matrices = make_random_adj_matrices(num_matrices=500, matrix_shape=(states, states))
    print(len(test_matrices))
    for idx1, m1 in enumerate(test_matrices):
        for idx2, m2 in enumerate(test_matrices[idx1 + 1:]):
            m_diff = m1 - m2
            frobenius_norm = np.linalg.norm(m_diff, 'fro')
            test_stats.append(frobenius_norm)
    # The 0.05 percentile for significance
    first_percentile = np.percentile(test_stats, 5)
    last_percentile = np.percentile(test_stats, 95)


    # calculate frobenius norms between the empirical transition matrices
    m_diff = M2 - calculated_M2
    frobenius_norm_M2 = np.linalg.norm(m_diff, 'fro')
    m_diff = M3 - calculated_M3
    frobenius_norm_M3 = np.linalg.norm(m_diff, 'fro')
    m_diff = M4 - calculated_M4
    frobenius_norm_M4 = np.linalg.norm(m_diff, 'fro')

    #frobenius_norms = [frobenius_norm_M2, frobenius_norm_M3]

    # plot all the results
    #print(test_stats)
    if True:
        ax.hist(test_stats, bins='auto', edgecolor='black')  # Adjust the number of bins as needed
        ax.axvline(0, color='orange', label='True Norm')
        ax.axvline(frobenius_norm_M2, color='blue', label='Frobenius Norm M2')
        ax.axvline(frobenius_norm_M3, color='cyan', label='Frobenius Norm M3')
        ax.axvline(frobenius_norm_M4, color='green', label='Frobenius Norm M4')
        #plt.axvline(f, color='green', label='Frobenius Norm between chunks')
        ax.axvline(first_percentile, color='red', label='First 0.05 percentile', linestyle='--')
        ax.axvline(last_percentile, color='red', label='First 0.95 percentile', linestyle='--')

        #ax.set_xlabel('Values')
        #ax.set_ylabel('Frequency')
        ax.set_title(f'{states} States')
        if states == 1:
            ax.legend()  # Display the legend
        ax.grid(True)
        #plt.show()
    return ax

#true_seq = generate_markov_process(M=3000, N=4, order=1)
#two_step_test(true_seq)




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

data = Database(worm_num, verbose=1)
data.exclude_neurons(b_neurons)

# Adding prediction Model & Cluster BPT
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
data.fit_model(logreg, binary=True)
nrcluster = 5
data.cluster_BPT(nrep=10, max_clusters=nrcluster, plot_markov=True, sim_markov=500)


exit()
xc_clusters = data.xc[:, :, 0].astype(int)

fig, ax = plt.subplots(4,6)

for cl in range(nrcluster):
    x = cl%6
    y = int(np.floor(cl/6))
    xctmp = xc_clusters[:, cl]

    two_step_test(xctmp, ax[y, x])
plt.show()
#for r in range(5):
#    xctmp = all_xc_3_clusters[:, r].astype(int)
#    p, _ = markovian(xctmp, K=200)
#    _, p_stationary = test_stationarity(xctmp, parts=2, plot=True)


