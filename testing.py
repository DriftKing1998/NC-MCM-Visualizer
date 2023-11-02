import matplotlib.pyplot as plt
import numpy as np

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sample data points
x = np.asarray([1, 2, 3, 4, 5, 5, 6]).astype(float)
y = np.asarray([2, 3, 4, 5, 6, 9, 14]).astype(float)
z = np.asarray([1, 2, 3, 4, 5, 7, 8]).astype(float)

print(np.arctan2(x, y))

# Calculate the direction vectors for the quivers
dx = np.diff(x)  # Differences between x coordinates
dy = np.diff(y)  # Differences between y coordinates
dz = np.diff(z)  # Differences between z coordinates

# Normalize the direction vectors
length = np.sqrt(dx**2 + dy**2 + dz**2)
dx /= length
dy /= length
dz /= length

# Create the quiver plot
ax.quiver(x[:-1], y[:-1], z[:-1], dx, dy, dz, length=1, normalize=True, color=['red', 'red', 'red', 'blue', 'red', 'blue', 'red', 'blue'])

# Scatterplot the points
#ax.scatter(x, y, z, c='r', marker='o', s=50, label='Points')

# Customize the plot as needed
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.legend()
plt.show()
