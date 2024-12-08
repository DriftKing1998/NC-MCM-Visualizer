import pandas as pd
from matplotlib.lines import Line2D
from pandas._typing import IndexLabel

import ncmcm as nc
import matplotlib.pyplot as plt
import numpy as np

states = 3
lengths = [1000, 2000, 4000, 8000]
sims = 150
alpha = 0.05

orders = {'Markov Process of 1st order': 1,
          'Random Process': 0,
          'Markov Process of 5th order': 5,
          'Markov Process of 4th order': 4,
          'Markov Process of 3rd order': 3,
          'Markov Process of 2nd order': 2}
colors = ['red', 'blue', 'green', 'orange', 'olive', 'salmon']
m = ['.', 'o', 'p', 'D', '^', '|']
cycles = 50
e_s = []
rej = []

for idx_l, l in enumerate(lengths):
    rejections = np.zeros((len(orders), cycles))
    variances = np.zeros((len(orders), cycles))
    variances_simulated = np.zeros((len(orders), cycles * sims))
    for y in range(cycles):
        print(f'Cycle No#{y + 1}')
        for x, order in enumerate(orders):
            if orders[order] == 0:
                seq = nc.simulate_random_sequence(N=states, M=l)
                print(f'Created a Random sequence')
            else:
                seq, _ = nc.simulate_markov_sequence(N=states, M=l, order=orders[order])
                print(f'Created a Markov sequence of order {orders[order]}')

            p, _, seq_var, seq_vars = nc.markov_property_test(seq, simulations=sims, return_variances=True)

            variances[x, y] = seq_var
            variances_simulated[x, y * sims:(y + 1) * sims] = seq_vars
            rejections[x, y] = 1 if p < alpha else 0

    effect_sizes = []

    for i in range(variances.shape[0]):
        mean_1st = np.mean(variances_simulated[i, :])
        var_1st = np.var(variances_simulated[i, :])

        mean_current = np.mean(variances[i, :])
        var_current = np.var(variances[i, :])

        a = var_1st * (cycles - 1)
        b = var_current * (cycles * sims - 1)
        c = cycles * (1 + sims) - 2

        pooled_std = np.sqrt((a + b) / c)
        cohen_d = (mean_current - mean_1st) / pooled_std
        effect_sizes.append(cohen_d)

    e_s.append(effect_sizes)
    rej.append(rejections)

# SAVING DATA
effect_sizes = np.asarray(e_s)
rejections = np.asarray(rej)
indexes = [f'{list(orders.keys())[i]}' for i in range(effect_sizes.shape[1])]

with pd.ExcelWriter(f"Markov_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    df = pd.DataFrame(effect_sizes[:, :].T, index=indexes)  # Convert the 2D slice to a DataFrame
    df.to_excel(writer,
                index=True,
                header=[f'{lengths[i]}' for i in range(effect_sizes.shape[0])])

with pd.ExcelWriter(f"Markov_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(rejections.shape[0]):
        df = pd.DataFrame(rejections[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(rejections.shape[1])])

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlabel('Effect Size [Cohen\'s D]', fontsize=12)
ax.set_ylabel('Power [1-ß]', fontsize=12)
ax.set_title(f'Power Curve for different sequence lengths and {states} states', fontsize=14)
# plt.suptitle(f'Ordering from right to left: 2nd order, 3rd order, 4th order, 5th order and random Process',
#              fontsize=8)

for idx_l, l in enumerate(lengths):
    effect_sizes = e_s[idx_l]
    rejections = rej[idx_l]

    for o in range(len(orders)):
        ax.scatter(effect_sizes[o], np.mean(rejections[o, :]),
                   color=colors[idx_l],
                   marker=m[o],
                   linestyle='-')
    ax.plot(effect_sizes, np.mean(rejections[:, :], axis=1),
            color=colors[idx_l],
            linestyle='-')

# Customize x-axis ticks to only show specific lengths
ax.axvline(0, linestyle='--', linewidth=0.5, color='black')
color_legend = [Line2D([0], [0], color=color, lw=4, label=f'Length {l}')
                for color, l in zip(colors[:len(lengths)], lengths)]
marker_legend = [Line2D([0], [0], marker=mark, color='black', lw=0, markersize=8, label=f'{order}')
                 for mark, order in zip(m, list(orders.keys())[:])]
legend1 = ax.legend(handles=color_legend, title="Lengths", loc='upper left')
ax.add_artist(legend1)
ax.legend(handles=marker_legend, title="Orders", loc='lower right')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Markov_{states}_{cycles}_{sims}.png')
plt.close()
#plt.show()

######
# 5 #
######

states = 5
lengths = [1000, 2000, 4000, 8000]
sims = 150
alpha = 0.05

orders = {'Markov Process of 1st order': 1,
          'Random Process': 0,
          'Markov Process of 5th order': 5,
          'Markov Process of 4th order': 4,
          'Markov Process of 3rd order': 3,
          'Markov Process of 2nd order': 2}
colors = ['red', 'blue', 'green', 'orange', 'olive', 'salmon']
m = ['.', 'o', 'p', 'D', '^', '|']
cycles = 50
e_s = []
rej = []

for idx_l, l in enumerate(lengths):
    rejections = np.zeros((len(orders), cycles))
    variances = np.zeros((len(orders), cycles))
    variances_simulated = np.zeros((len(orders), cycles * sims))
    for y in range(cycles):
        print(f'Cycle No#{y + 1}')
        for x, order in enumerate(orders):
            if orders[order] == 0:
                seq = nc.simulate_random_sequence(N=states, M=l)
                print(f'Created a Random sequence')
            else:
                seq, _ = nc.simulate_markov_sequence(N=states, M=l, order=orders[order])
                print(f'Created a Markov sequence of order {orders[order]}')

            p, _, seq_var, seq_vars = nc.markov_property_test(seq, simulations=sims, return_variances=True)

            variances[x, y] = seq_var
            variances_simulated[x, y * sims:(y + 1) * sims] = seq_vars
            rejections[x, y] = 1 if p < alpha else 0

    effect_sizes = []

    for i in range(variances.shape[0]):
        mean_1st = np.mean(variances_simulated[i, :])
        var_1st = np.var(variances_simulated[i, :])

        mean_current = np.mean(variances[i, :])
        var_current = np.var(variances[i, :])

        a = var_1st * (cycles - 1)
        b = var_current * (cycles * sims - 1)
        c = cycles * (1 + sims) - 2

        pooled_std = np.sqrt((a + b) / c)
        cohen_d = (mean_current - mean_1st) / pooled_std
        effect_sizes.append(cohen_d)

    e_s.append(effect_sizes)
    rej.append(rejections)

# SAVING DATA
effect_sizes = np.asarray(e_s)
rejections = np.asarray(rej)
indexes = [f'{list(orders.keys())[i]}' for i in range(effect_sizes.shape[1])]

with pd.ExcelWriter(f"Markov_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    df = pd.DataFrame(effect_sizes[:, :].T, index=indexes)  # Convert the 2D slice to a DataFrame
    df.to_excel(writer,
                index=True,
                header=[f'{lengths[i]}' for i in range(effect_sizes.shape[0])])

with pd.ExcelWriter(f"Markov_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(rejections.shape[0]):
        df = pd.DataFrame(rejections[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(rejections.shape[1])])

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlabel('Effect Size [Cohen\'s D]', fontsize=12)
ax.set_ylabel('Power [1-ß]', fontsize=12)
ax.set_title(f'Power Curve for different sequence lengths and {states} states', fontsize=14)
# plt.suptitle(f'Ordering from right to left: 2nd order, 3rd order, 4th order, 5th order and random Process',
#              fontsize=8)

for idx_l, l in enumerate(lengths):
    effect_sizes = e_s[idx_l]
    rejections = rej[idx_l]

    for o in range(len(orders)):
        ax.scatter(effect_sizes[o], np.mean(rejections[o, :]),
                   color=colors[idx_l],
                   marker=m[o],
                   linestyle='-')
    ax.plot(effect_sizes, np.mean(rejections[:, :], axis=1),
            color=colors[idx_l],
            linestyle='-')

# Customize x-axis ticks to only show specific lengths
ax.axvline(0, linestyle='--', linewidth=0.5, color='black')
color_legend = [Line2D([0], [0], color=color, lw=4, label=f'Length {l}')
                for color, l in zip(colors[:len(lengths)], lengths)]
marker_legend = [Line2D([0], [0], marker=mark, color='black', lw=0, markersize=8, label=f'{order}')
                 for mark, order in zip(m, list(orders.keys())[:])]
legend1 = ax.legend(handles=color_legend, title="Lengths", loc='upper left')
ax.add_artist(legend1)
ax.legend(handles=marker_legend, title="Orders", loc='lower right')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Markov_{states}_{cycles}_{sims}.png')
plt.close()

######
# 10 #
######

states = 10
lengths = [1000, 2000, 4000, 8000]
sims = 150
alpha = 0.05

orders = {'Markov Process of 1st order': 1,
          'Random Process': 0,
          'Markov Process of 5th order': 5,
          'Markov Process of 4th order': 4,
          'Markov Process of 3rd order': 3,
          'Markov Process of 2nd order': 2}
colors = ['red', 'blue', 'green', 'orange', 'olive', 'salmon']
m = ['.', 'o', 'p', 'D', '^', '|']
cycles = 50
e_s = []
rej = []

for idx_l, l in enumerate(lengths):
    rejections = np.zeros((len(orders), cycles))
    variances = np.zeros((len(orders), cycles))
    variances_simulated = np.zeros((len(orders), cycles * sims))
    for y in range(cycles):
        print(f'Cycle No#{y + 1}')
        for x, order in enumerate(orders):
            if orders[order] == 0:
                seq = nc.simulate_random_sequence(N=states, M=l)
                print(f'Created a Random sequence')
            else:
                seq, _ = nc.simulate_markov_sequence(N=states, M=l, order=orders[order])
                print(f'Created a Markov sequence of order {orders[order]}')

            p, _, seq_var, seq_vars = nc.markov_property_test(seq, simulations=sims, return_variances=True)

            variances[x, y] = seq_var
            variances_simulated[x, y * sims:(y + 1) * sims] = seq_vars
            rejections[x, y] = 1 if p < alpha else 0

    effect_sizes = []

    for i in range(variances.shape[0]):
        mean_1st = np.mean(variances_simulated[i, :])
        var_1st = np.var(variances_simulated[i, :])

        mean_current = np.mean(variances[i, :])
        var_current = np.var(variances[i, :])

        a = var_1st * (cycles - 1)
        b = var_current * (cycles * sims - 1)
        c = cycles * (1 + sims) - 2

        pooled_std = np.sqrt((a + b) / c)
        cohen_d = (mean_current - mean_1st) / pooled_std
        effect_sizes.append(cohen_d)

    e_s.append(effect_sizes)
    rej.append(rejections)

# SAVING DATA
effect_sizes = np.asarray(e_s)
rejections = np.asarray(rej)
indexes = [f'{list(orders.keys())[i]}' for i in range(effect_sizes.shape[1])]

with pd.ExcelWriter(f"Markov_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    df = pd.DataFrame(effect_sizes[:, :].T, index=indexes)  # Convert the 2D slice to a DataFrame
    df.to_excel(writer,
                index=True,
                header=[f'{lengths[i]}' for i in range(effect_sizes.shape[0])])

with pd.ExcelWriter(f"Markov_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(rejections.shape[0]):
        df = pd.DataFrame(rejections[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(rejections.shape[1])])

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlabel('Effect Size [Cohen\'s D]', fontsize=12)
ax.set_ylabel('Power [1-ß]', fontsize=12)
ax.set_title(f'Power Curve for different sequence lengths and {states} states', fontsize=14)
# plt.suptitle(f'Ordering from right to left: 2nd order, 3rd order, 4th order, 5th order and random Process',
#              fontsize=8)

for idx_l, l in enumerate(lengths):
    effect_sizes = e_s[idx_l]
    rejections = rej[idx_l]

    for o in range(len(orders)):
        ax.scatter(effect_sizes[o], np.mean(rejections[o, :]),
                   color=colors[idx_l],
                   marker=m[o],
                   linestyle='-')
    ax.plot(effect_sizes, np.mean(rejections[:, :], axis=1),
            color=colors[idx_l],
            linestyle='-')

# Customize x-axis ticks to only show specific lengths
ax.axvline(0, linestyle='--', linewidth=0.5, color='black')
color_legend = [Line2D([0], [0], color=color, lw=4, label=f'Length {l}')
                for color, l in zip(colors[:len(lengths)], lengths)]
marker_legend = [Line2D([0], [0], marker=mark, color='black', lw=0, markersize=8, label=f'{order}')
                 for mark, order in zip(m, list(orders.keys())[:])]
legend1 = ax.legend(handles=color_legend, title="Lengths", loc='upper left')
ax.add_artist(legend1)
ax.legend(handles=marker_legend, title="Orders", loc='lower right')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Markov_{states}_{cycles}_{sims}.png')
plt.close()
