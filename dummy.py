from scripts.classes import *

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


n_reps = 3
n_clusters = 15
p_m = np.zeros((n_clusters, n_reps, 5))

for i in range(5):
    data = Database(data_set_no=i)
    data.exclude_neurons(b_neurons)
    s = set({})
    for idx1, i1 in enumerate(data.B):
        for i2 in data.B[idx1:]:
            if i1 != i2:
                s.add((i1, i2))

    print(s)
    print(len(s))
    #logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)

    #data.fit_model(logreg, markov_test=True, nrep=n_reps, max_clusters=n_clusters, sim_markov=10)
    #p_m[:, :, i] = data.p_markov

plot_markov = np.mean(p_m, axis=1).T

average_markov_plot(plot_markov)


