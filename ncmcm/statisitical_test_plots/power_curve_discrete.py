import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

import ncmcm as nc
import matplotlib.pyplot as plt
import numpy as np

states = 10
lengths = [1000, 2000, 4000, 8000]
sims = 50
alpha = 0.05
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
x_ticks = [0, 0.25, 0.5, 0.75, 1]
stds = np.zeros((6, 4))

orders = {'Markov Process of 1st order': 1,
          'Random Process': 0,
          '200 changes': 200,
          '100 changes': 100,
          '50 changes': 50,
          '5 changes': 5}
colors = ['red', 'blue', 'green', 'orange', 'salmon', 'olive', 'black']
m = ['.', 'o', 'p', 'D', '^', '|']

cycles = 5
eff_ks = []
eff_tt = []
rej_ks = []
rej_tt = []

for idx_l, l in enumerate(lengths):
    print(f'LENGTH: {l}')
    rejections_ks = np.zeros((len(orders), cycles))
    rejections_tt = np.zeros((len(orders), cycles))
    effect_ks = np.zeros((len(orders), cycles))
    effect_tt = np.zeros((len(orders), cycles))
    for y in range(cycles):
        print(f'Cycle No#{y + 1}')
        for x, order in enumerate(orders):
            if orders[order] == 0:
                seq = nc.simulate_random_sequence(N=states, M=l)
            else:
                if orders[order] == 1:
                    seq, _ = nc.simulate_markov_sequence(N=states, M=l, order=1)
                else:
                    seq = nc.discrete_non_stationary_process(N=states, M=l, changes=orders[order])

            p_ks, ks, p_t, tt = nc.stationary_property_test(seq,
                                                            simulations=sims,
                                                            num_states=states,
                                                            chunks_num=5,
                                                            test_mode='both')
            effect_ks[x, y] = ks
            effect_tt[x, y] = tt
            rejections_ks[x, y] = 1 if p_ks < alpha else 0
            rejections_tt[x, y] = 1 if p_t < alpha else 0

    eff_ks.append(effect_ks)
    eff_tt.append(effect_tt)
    rej_ks.append(rejections_ks)
    rej_tt.append(rejections_tt)

# SAVING DATA
effect_sizes_tt = np.asarray(eff_tt)
effect_sizes_ks = np.asarray(eff_ks)
rejections_tt = np.asarray(rej_tt)
rejections_ks = np.asarray(rej_ks)

with pd.ExcelWriter(f"StationaryChanges_TT_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(effect_sizes_tt.shape[0]):
        df = pd.DataFrame(effect_sizes_tt[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(effect_sizes_tt.shape[1])])
with pd.ExcelWriter(f"StationaryChanges_KS_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(effect_sizes_ks.shape[0]):
        df = pd.DataFrame(effect_sizes_ks[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(effect_sizes_ks.shape[1])])
with pd.ExcelWriter(f"StationaryChanges_TT_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(rejections_tt.shape[0]):
        df = pd.DataFrame(rejections_tt[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(rejections_tt.shape[1])])
with pd.ExcelWriter(f"StationaryChanges_KS_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    for i in range(rejections_ks.shape[0]):
        df = pd.DataFrame(rejections_ks[i, :, :].T)  # Convert the 2D slice to a DataFrame
        df.to_excel(writer,
                    sheet_name=f"{lengths[i]}",
                    index=False,
                    header=[f'{list(orders.keys())[i]}' for i in range(rejections_ks.shape[1])])

# PLOTTING DATA
fig, ax = plt.subplots(ncols=len(lengths) + 1, nrows=2, figsize=(14, 8))
fig.suptitle(f'Power Plots for different sequences of {states} states', fontsize=14, fontweight='bold')

for axes in range(2):
    for idx_l, l in enumerate(lengths):
        effect_sizes = [eff_ks[idx_l] if axes == 0 else eff_tt[idx_l]][0]  # e_s[idx_l]
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
# legend1 = ax[axes, l].legend(handles=color_legend, title="Lengths", loc='upper left')
# ax[axes, l].add_artist(legend1)
# ax[0, 4].legend(handles=marker_legend, title="Orders", loc='lower center')

legend1 = ax[0, len(lengths)].legend(
    handles=color_legend,
    title="Sequence Lengths",title_fontsize=15,
    loc='lower left',
    bbox_to_anchor=(0, 0.1),
    labelspacing=1.2
)

legend2 = ax[1, len(lengths)].legend(
    handles=marker_legend,
    title="Orders",title_fontsize=15,
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
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Changes_{states}_{cycles}_{sims}.png')
plt.close()

with pd.ExcelWriter(f"StationaryChanges_KS_STD_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
    df = pd.DataFrame(stds[:, :], index=list(orders.keys()))  # Convert the 2D slice to a DataFrame
    df.to_excel(writer,
                index=True,
                header=[f'{lengths[i]}' for i in range(stds.shape[1])])


for i in range(6):
    vals = stds[i, :].reshape(-1, 1)
    lin = LinearRegression()
    lin.fit(vals, y=[j+1 for j in range(4)])
    print(lin.coef_)
