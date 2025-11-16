from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary_with_pca(model, X, y):
    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create meshgrid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid)

    # Predict on original (non-PCA) grid
    Z = model.predict(grid_original)
    Z = Z.reshape(xx.shape)

    # Colormaps
    background_colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700']
    point_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500']
    cmap_background = ListedColormap(background_colors[:len(np.unique(y))])
    cmap_points = ListedColormap(point_colors[:len(np.unique(y))])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_points, edgecolors='k')
    plt.title("Decision Boundary with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
