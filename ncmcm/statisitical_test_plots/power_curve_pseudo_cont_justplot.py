import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

import ncmcm as nc
import matplotlib.pyplot as plt
import numpy as np

states = 5
lengths = [1000, 2000, 4000, 8000]
sims = 50
alpha = 0.05
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
x_ticks = [0, 0.25, 0.5, 0.75, 1]
stds = np.zeros((6, 4))

orders = {'Markov Process of 1st order': 1,
          'Random Process': 0,
          '5 changes with epsilon 0.02': 5,
          '5 changes with epsilon 0.04': 4,
          '5 changes with epsilon 0.06': 3,
          '5 changes with epsilon 0.08': 2}
colors = ['red', 'blue', 'green', 'orange', 'olive', 'salmon']  # , 'black']
m = ['.', 'o', 'p', 'D', '^', '|']

cycles = 50

res_1 = np.zeros((4, 6, 50))
res = pd.read_excel(f"StationaryEpsilon_TT_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    res_1[i, :, :] = res[r].T

res_2 = np.zeros((4, 6, 50))
res = pd.read_excel(f"StationaryEpsilon_TT_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    res_2[i, :, :] = res[r].T

res_3 = np.zeros((4, 6, 50))
res = pd.read_excel(f"StationaryEpsilon_KS_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    res_3[i, :, :] = res[r].T

res_4 = np.zeros((4, 6, 50))
res = pd.read_excel(f"StationaryEpsilon_KS_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    res_4[i, :, :] = res[r].T


eff_tt = np.asarray(res_1)
rej_tt = np.asarray(res_2)
eff_ks = np.asarray(res_3)
rej_ks = np.asarray(res_4)

print(eff_tt[0, :, :5])
print(rej_tt[0, :, :5])

# PLOTTING DATA
fig, ax = plt.subplots(ncols=len(lengths) + 1, nrows=2, figsize=(14, 8))
fig.suptitle(f'Power Plots for different sequences of {states} states', fontsize=14, fontweight='bold')

for axes in range(2):
    for idx_l, l in enumerate(lengths):
        effect_sizes = [eff_ks[idx_l] if axes == 0 else eff_tt[idx_l]][0]
        rejections = [rej_ks[idx_l] if axes == 0 else rej_tt[idx_l]][0]

        if axes == 1:
            ax[axes, idx_l].set_xlabel('TT-Statistic', fontsize=12)
            if idx_l == 0:
                ax[axes, idx_l].set_ylabel('Power [1-ß]', fontsize=12)
        else:
            ax[axes, idx_l].set_xlabel('KS-Statistic', fontsize=12)
            if idx_l == 0:
                ax[axes, idx_l].set_ylabel('Power [1-ß]', fontsize=12)

        for o in range(len(orders)):
            if axes == 0:
                stds[o, idx_l] = np.std(effect_sizes[o, :])
            for u in range(rejections.shape[1]):
                ax[axes, idx_l].scatter(effect_sizes[o, u],
                                        np.mean(rejections[o, :]),
                                        label=f'Length {l}',
                                        color=colors[idx_l],
                                        marker=m[o],
                                        alpha=0.05)

    eff_ks_tmp = np.mean(np.asarray(eff_ks), axis=2)
    eff_tt_tmp = np.mean(np.asarray(eff_tt), axis=2)
    rej_ks_tmp = np.mean(np.asarray(rej_ks), axis=2)
    rej_tt_tmp = np.mean(np.asarray(rej_tt), axis=2)
    print('point at: ', eff_ks_tmp, rej_ks_tmp)
    print('point at: ', eff_tt_tmp, rej_tt_tmp)
    x_line = [eff_ks_tmp if axes == 0 else eff_tt_tmp][0]
    y_line = [rej_ks_tmp if axes == 0 else rej_tt_tmp][0]
    for l in range(eff_tt_tmp.shape[0]):
        ax[axes, l].plot(x_line[l, :], y_line[l, :],
                         color=colors[l],
                         linestyle='-')
        for i in range(len(x_line[l, :])):
            ax[axes, l].scatter(
                x_line[l, i], y_line[l, i],
                color=colors[l],
                marker=m[i],
                linestyle='-'
            )
        ax[axes, l].axvline(0, linestyle='--', linewidth=3, color='black')
        print(f'Row-axes {axes}\nCol-axes {l}')
        ax[axes, l].set_ylim(-0.1, 1.1)
        if l == 0:
            ax[axes, l].set_yticks(y_ticks)
            ax[axes, l].set_yticklabels(np.asarray(y_ticks).astype(str))
        else:
            ax[axes, l].set_yticks(y_ticks)
            ax[axes, l].set_yticklabels([])
        if axes == 0:

            ax[axes, l].set_xticks(x_ticks)
            ax[axes, l].set_xticklabels(np.asarray(x_ticks).astype(str))

color_legend = [Line2D([0], [0], color=color, lw=4, label=f'Length {l}')
                for color, l in zip(colors[:len(lengths)], lengths)]
marker_legend = [Line2D([0], [0], marker=mark, color='black', lw=0, markersize=8, label=f'{order}')
                 for mark, order in zip(m, list(orders.keys()))]

legend1 = ax[0, len(lengths)].legend(
    handles=color_legend,
    title="Sequence Lengths", title_fontsize=15,
    loc='lower left',
    bbox_to_anchor=(0, 0.1),
    labelspacing=1.2
)

legend2 = ax[1, len(lengths)].legend(
    handles=marker_legend,
    title="Orders", title_fontsize=15,
    loc='lower left',
    bbox_to_anchor=(0, 0.1),
    labelspacing=1.2
)
ax[0, len(lengths)].add_artist(legend1)
ax[1, len(lengths)].add_artist(legend2)

ax[0, len(lengths)].grid(False)
ax[1, len(lengths)].grid(False)
ax[0, len(lengths)].set_xticks([])
ax[0, len(lengths)].set_yticks([])
ax[1, len(lengths)].set_xticks([])
ax[1, len(lengths)].set_yticks([])
for spine in ax[0, len(lengths)].spines.values():
    spine.set_visible(False)
for spine in ax[1, len(lengths)].spines.values():
    spine.set_visible(False)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Epsilon_{states}_{cycles}_{sims}.png')
plt.show()
exit()
