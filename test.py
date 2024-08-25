import ncmcm as nc
import matplotlib.pyplot as plt
import numpy as np

states = 10
l = np.array([100, 500, 1000, 2000, 5000, 10000, 20000])
orders = ['Markov Process of 1st order',
          'Markov Process of 2nd order',
          'Markov Process of 3rd order',
          'Markov Process of 4th order',
          'Markov Process of 5th order',
          'Random Process']
colors = ['red', 'blue', 'green', 'orange', 'salmon', 'olive']
cycles = 30
offsets = np.linspace(-7, 7, len(orders))
tmp = np.asarray([
    [1.0, 0.06666666666666667, 0.0, 0.03333333333333333, 0.03333333333333333, 0.0, 0.0],
    [1.0, 0.0, 0.26666666666666666, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.06666666666666667, 0.9333333333333333, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.03333333333333333, 0.16666666666666666, 0.4],
    [1.0, 0.0, 0.0, 0.0, 0.1, 0.03333333333333333, 0.13333333333333333],
    [1.0, 0.0, 0.0, 0.03333333333333333, 0.03333333333333333, 0.13333333333333333, 0.03333333333333333]])
# tmp = np.asarray([
#     [0.03333333333333333, 0.0, 0.03333333333333333, 0.0, 0.03333333333333333, 0.0],
#     [0.3, 1.0, 1.0, 1.0, 1.0, 1.0],
#     [0.0, 0.6, 0.8, 1.0, 1.0, 1.0],
#     [0.0, 0.03333333333333333, 0.23333333333333334, 0.4, 0.9333333333333333, 1.0],
#     [0.03333333333333333, 0.06666666666666667, 0.03333333333333333, 0.06666666666666667, 0.1, 0.5],
#     [0.0, 0.03333333333333333, 0.06666666666666667, 0.03333333333333333, 0.03333333333333333, 0.1]
# ])
plt.figure(figsize=(12, 8))

for _ in range(tmp.shape[0]):
    results = tmp[_, :]
    shifted_l = l + offsets[_]
    plt.plot(shifted_l, results, label=orders[_], color=colors[_], marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Power [1-ß]', fontsize=12)
plt.title('Power Table', fontsize=14)

# Customize x-axis ticks to only show specific lengths
plt.xticks(l)
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
