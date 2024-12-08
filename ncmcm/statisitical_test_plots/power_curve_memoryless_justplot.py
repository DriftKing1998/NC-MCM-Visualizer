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



res_1 = np.zeros((4, 6))
res = pd.read_excel(f"Markov_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None, index_col=0)
for i, r in enumerate(res):
    print(type(res[r]), res[r].shape)
    print(res[r])
    res_1[:, :] = res[r].T

res_2 = np.zeros((4, 6, 50))
res = pd.read_excel(f"Markov_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl", sheet_name=None)
for i, r in enumerate(res):
    res_2[i, :, :] = res[r].T


e_s = np.asarray(res_1)
rej = np.asarray(res_2)


# SAVING DATA
effect_sizes = np.asarray(e_s)
rejections = np.asarray(rej)
# indexes = [f'{list(orders.keys())[i]}' for i in range(effect_sizes.shape[1])]
#
# with pd.ExcelWriter(f"Markov_ES_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
#     df = pd.DataFrame(effect_sizes[:, :].T, index=indexes)  # Convert the 2D slice to a DataFrame
#     df.to_excel(writer,
#                 index=True,
#                 header=[f'{lengths[i]}' for i in range(effect_sizes.shape[0])])
#
# with pd.ExcelWriter(f"Markov_REJ_{states}_{cycles}_{sims}.xlsx", engine="openpyxl") as writer:
#     for i in range(rejections.shape[0]):
#         df = pd.DataFrame(rejections[i, :, :].T)  # Convert the 2D slice to a DataFrame
#         df.to_excel(writer,
#                     sheet_name=f"{lengths[i]}",
#                     index=False,
#                     header=[f'{list(orders.keys())[i]}' for i in range(rejections.shape[1])])

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
    print(f'Length {l}:')
    print(rejections)

    for o in range(len(orders)):
        print(f'MEAN REJECTIONS: {np.mean(rejections[o, :])} for {list(orders.keys())[o]}')
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
#plt.savefig(f'Markov_{states}_{cycles}_{sims}.png')
#plt.close()
plt.show()
