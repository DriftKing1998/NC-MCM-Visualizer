import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.decomposition import PCA, NMF

import numpy as np

X, color = datasets._samples_generator.make_swiss_roll(n_samples=5000)
print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
print("Done. Reconstruction error: %g" % err)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(X.shape, X_r.shape, X_pca.shape, color.shape)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

# Scatter plot in 3D
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
# Scatter plot in 2D
ax2.scatter(X_r[:, 0], X_r[:, 1], c=color)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=color)

ax1.grid(False)
ax1.set_title('Original')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
#ax1.set_axis_off()
ax2.grid(False)
ax2.set_title('LLE')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax3.grid(False)
ax3.set_title('PCA')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
plt.show()