def ordinal(n):
    return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


from ncmcm.ncmcm_classes.Database import *

# Define parameters
alpha = 0.05  # significance level
markov_orders = [1, 2, 3, 4, 5]#, 6, 7, 8]# [2, 3, 4, 5, 6]  # effect sizes to test
sample_sizes = [1000, 5000, 10000, 20000]#, 10000]#500, 1000, 1500, 2000, 5000, 10000]  # range of sample sizes
states = 5  # 5
sm = 500
epsilon = 1e-8

effect_size = {}
stds = {}
means = {}
for m_o in markov_orders:
    for s in sample_sizes:
        TH0 = np.zeros(sm)
        for kperm in range(sm):
            zH0, _ = simulate_markovian(M=s, N=states, order=m_o)
            Pz0z1z2H0 = np.zeros((states, states, states))
            for m in range(2, s):
                i = zH0[m]  # col
                j = zH0[m - 1]  # row
                k = zH0[m - 2]  # depth
                Pz0z1z2H0[k, j, i] += 1

            Pz0z1z2H0 = Pz0z1z2H0 / (s - 2)
            Pz1z2H0 = np.sum(Pz0z1z2H0, axis=2)

            Pz1z2H0 = np.where(Pz1z2H0 == 0, epsilon, Pz1z2H0)
            Pz1z2H0 = Pz1z2H0 / np.sum(Pz1z2H0)

            P2H0 = Pz0z1z2H0 / np.tile(Pz1z2H0[:, :, np.newaxis], (1, 1, states))
            TH0[kperm] = np.sum(np.var(P2H0, axis=0).flatten())

        #plt.hist(TH0, bins=30)
        #plt.xlim(0, 0.5)
        #plt.show()

        means[f'{m_o}_order_{s}_ss'] = TH0
        stds[f'{m_o}_order_{s}_ss'] = np.std(TH0)
        print(
            f"For order {m_o} and sample size {s} we have:\nSTD: {stds[f'{m_o}_order_{s}_ss']}\nMEAN: {np.mean(means[f'{m_o}_order_{s}_ss'])}")

        if m_o == 1:
            continue
        else:

            # percentile_95_control = np.percentile(means[f'1_order_{s}_ss'], 5)
            #
            # effect_size[f'{m_o}_order_{s}_ss'] = (abs(np.mean(means[f'{m_o}_order_{s}_ss']) - percentile_95_control) /
            #                                       stds[f'1_order_{s}_ss'])

            effect_size[f'{m_o}_order_{s}_ss'] = (abs(np.mean(means[f'{m_o}_order_{s}_ss']) - np.mean(means[f'1_order_{s}_ss'])) /
                                                  stds[f'1_order_{s}_ss'])

for e, i in effect_size.items():
    if int(e.split('_')[2]) == 1000:
        print(e, i)
print()
for e, i in effect_size.items():
    if int(e.split('_')[2]) == 5000:
        print(e, i)
print()
for e, i in effect_size.items():
    if int(e.split('_')[2]) == 10000:
        print(e, i)
print()
for e, i in effect_size.items():
    if int(e.split('_')[2]) == 20000:
        print(e, i)


exit()


# Calculate power for each combination of sample size and effect size
power_results = []
table = False

if table:
    sims = 100
    seq_len = 5000
    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    for i, markov_order in enumerate(markov_orders):
        #print(i, markov_order)
        res = []
        simulated_sequences_false = [simulate_markovian(M=seq_len, N=states, order=markov_order)[0] for _ in range(sims)]
        simulated_sequences_true = [simulate_markovian(M=seq_len, N=states, order=1)[0] for _ in range(sims)]
        print('Simulated All')
        markovian_results_false = [markovian(s, sim_memoryless=300)[0] for s in simulated_sequences_false]
        print('Tested Falses')
        markovian_results_true = [markovian(s, sim_memoryless=300)[0] for s in simulated_sequences_true]
        print('Tested Trues')
        rejected_false = [alpha >= m_r_p for m_r_p in markovian_results_false]
        kept_false = [alpha < m_r_p for m_r_p in markovian_results_false]
        rejected_true = [alpha >= m_r_p for m_r_p in markovian_results_true]
        kept_true = [alpha < m_r_p for m_r_p in markovian_results_true]

        y = i % 2
        x = int(i / 2)

        cm = pd.DataFrame([[sum(rejected_false), sum(kept_false)],
                           [sum(rejected_true), sum(kept_true)]],
                          index=['HO False', 'HO True'],
                          columns=['H0 rejected', 'H0 kept'])
        annotation = np.array([['Correct', 'Type II error'], ['Type I error', 'Correct']])
        data = np.asarray(cm)
        formatted_text = (np.asarray(["{0}\n{1:.2f}%".format(text, (data / (sims*2)) * 100) for text, data in
                                      zip(annotation.flatten(), data.flatten())])).reshape(2, 2)

        heatmap(cm, annot=formatted_text, fmt="", ax=ax[x, y], cbar=False, cmap='Blues', center=sims/2)
        ax[x, y].set_title(
            f'{ordinal(markov_order)} order Markov process')
    #plt.suptitle(f'Sequence Length {seq_len} and {sims*2} simulations per plot')
    plt.tight_layout()
    plt.show()

    exit()


for markov_order in markov_orders:
    power_for_effect_size = []
    ms = []
    for n in sample_sizes:
        print(f'For markov order {markov_order} and sequence length {n} we reject:')
        simulated_sequences = [simulate_markovian(M=n, N=states, order=markov_order)[0] for _ in range(20)]
        markovian_results = [markovian(s, sim_memoryless=20)[0] for s in simulated_sequences]
        rejected = [alpha > m_r_p for m_r_p in markovian_results]
        print(sum(rejected), '/', len(rejected))
        power = sum(rejected) / 20  # proportion of rejections
        power_for_effect_size.append(power)
    power_results.append(power_for_effect_size)

power_for_random = []
for n in sample_sizes:
    print(f'For random process and sequence length {n} we reject:')
    simulated_sequences = [simulate_random_sequence(M=n, N=states) for _ in range(20)]
    markovian_results = [markovian(s, sim_memoryless=20)[0] for s in simulated_sequences]
    rejected = [alpha > m_r_p for m_r_p in markovian_results]
    print(sum(rejected), '/', len(rejected))
    power = sum(rejected) / 20  # proportion of rejections
    power_for_random.append(power)
power_results.append(power_for_random)

power_for_first = []
for n in sample_sizes:
    print(f'For 1st order Markov process and sequence length {n} we reject:')
    simulated_sequences = [simulate_markovian(M=n, N=states, order=1)[0] for _ in range(20)]
    markovian_results = [markovian(s, sim_memoryless=100)[0] for s in simulated_sequences]
    rejected = [alpha > m_r_p for m_r_p in markovian_results]
    print(sum(rejected), '/', len(rejected))
    power = sum(rejected) / 20  # proportion of rejections
    power_for_first.append(power)
power_results.append(power_for_first)

power_results = np.array(power_results)

# Plotting the power curve
fig, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=3)

for i, sample_size in enumerate(sample_sizes):
    x = i % 3
    y = int(i / 3)
    y_axis = list(power_results[:, i])
    y_axis.reverse()
    ax[y, x].plot([0, 1] + markov_orders, y_axis)
    ax[y, x].set_ylim(0, 1)
    ax[y, x].set_title(f'For Sequence Length {sample_size}')
    ax[y, x].set_xlabel('Effect size')
    ax[y, x].set_ylabel('Power')
    ax[y, x].grid(True)
    ax[y, x].set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax[y, x].set_xticklabels(['1st order', 'Random', '6th order', '5th order', '4th order', '3rd order', '2nd order'],
                             rotation=45)

fig.suptitle('Power Curve of Custom Statistical Test')
plt.tight_layout()
plt.show()
