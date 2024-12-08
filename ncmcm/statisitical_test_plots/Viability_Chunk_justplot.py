from matplotlib.lines import Line2D

import ncmcm as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



l = 3000
sims = 50
reps = 50
max_chunk = 15
max_states = 10  # -2
chunks_ticks = [n + 2 for n in range(max_chunk - 1)]
vocab = {0: ('1st order Markov', '#fe0000'),
         1: ('2nd order Markov', '#fe4499'),
         #2: ('3rd order Markov', 'orange'),
         #3: ('Random', 'yellow'),
         2: ('Ornstein-Uehlenbeck', '#fe998a'),
         3: ('Discrete Non-Stationary Markov', '#0000fe'),
         4: ('Pseudo-Cont. Non-Stationary Markov', '#4499fe'),
         5: ('Random Walk', '#998afe')}

fig, axes = plt.subplots(10, 1, figsize=(4, 18))
for x, N in enumerate(range(2, max_states, 2)):
    N = N + 18
    #x = row_col % 2
    #y = int(np.floor(row_col / 2))

    result = np.zeros((6, 50, 14))
    res = pd.read_excel(f"/Users/michaelhofer/Desktop/MS Data/ChunkViability_{N}_states.xlsx", engine="openpyxl", sheet_name=None)
    for i, r in enumerate(res):
        result[i, :, :] = res[r]
    print(result.shape)



    mean_values = np.mean(result, axis=1)
    ci_25 = np.percentile(result, 12.5, axis=1)
    ci_75 = np.percentile(result, 87.5, axis=1)
    print(f'SHAPE MEANS {mean_values.shape}')
    for i in range(result.shape[0]):  # Iterate over axis 0
        color = vocab[i][1]  # Get the corresponding color from vocab
        axes[x].plot(chunks_ticks, mean_values[i], label=f"{vocab[i][0]}", color=color)  # Line for the mean
        axes[x].fill_between(
            chunks_ticks,  # X-axis values
            ci_25[i], ci_75[i],  # Lower and upper bounds for CI
            color=color, alpha=0.15  # Shaded band
        )
    #if x == 0:

    axes[x].set_xticks(chunks_ticks)
    axes[x].set_xticklabels(chunks_ticks)
    if x == 9:
        axes[x].set_xlabel('Amount of Chunks')
    axes[x].set_ylabel('P-Values')
    axes[x].set_title(f'States {N}')
    #axes[x].set_xlabel('Amount of chunks')


# axes[10].grid(False)
# axes[10].set_axis_off()
#
# axes[10].set_xlabel(None)
# axes[10].set_xticks([])
# axes[10].set_yticks([])
# axes[10].set_xticklabels([])
# axes[10].set_yticklabels([])


axes[4].grid(False)
axes[4].set_axis_off()

axes[4].set_xlabel(None)
axes[4].set_xticks([])
axes[4].set_yticks([])
axes[4].set_xticklabels([])
axes[4].set_yticklabels([])


marker_legend = [Line2D([0], [0], marker='.', color=c, lw=0, markersize=8, label=f'{t} Process')
                 for t, c in vocab.values()]
# axes[10].legend(handles=marker_legend, title="Processes", loc='center')
axes[4].legend(handles=marker_legend, title="Processes", loc='center')
plt.tight_layout()
plt.savefig("LONG_chunks_2.png")  # Save the plot to a file
plt.show()
#plt.show()
