import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
np.random.seed(42)
num_points = 100
x = np.arange(num_points) / 440
y = np.random.rand(num_points) * 10  # Adjust the scale as needed

# Calculate arrow directions
dx = np.diff(x)
dy = np.diff(y)

# Create a quiver plot
fig, ax = plt.subplots(figsize=(9, 13))
ax.quiver(x[:-1], y[:-1], dx, dy, scale_units='xy', angles='xy', scale=1, color='blue')

# Scatter plot for the points
ax.scatter(x, y, color='red')

# Set axis limits
ax.set_xlim(x.min() - 1, x.max() + 1)
ax.set_ylim(y.min() - 1, y.max() + 1)

plt.show()
