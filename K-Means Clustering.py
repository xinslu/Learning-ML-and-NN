import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger


def initializeCentroids(X, K):
    randomidx = np.random.permutation(X.shape[0])
    centroids = X[randomidx[0:K]]
    return centroids


def plotCentroids(centroids, K, colors, markers):
    cmap = ListedColormap(colors[:K])
    for i in range(K):
        plt.scatter(centroids[:, [0]][i], centroids[:, [1]]
                    [i], c=cmap(i), marker=markers[i], s=100)


def closestCentroid(X, centroids, colors, K, markers):
    cmap = ListedColormap(colors[:K])
    idxarray = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        lowest, idx = ((centroids[0][0] - X[i][0]) **
                       2 + (centroids[0][1] - X[i][1]) ** 2), 0
        for j in range(1, centroids.shape[0]):
            distance = (centroids[j][0] - X[i][0])**2 + \
                (centroids[j][1] - X[i][1])**2
            if lowest > distance:
                lowest = distance
                idx = j
        idxarray[i] = idx
        plt.scatter(X[i][0], X[i][1], c=cmap(idx), marker=markers[idx], s=5)
    plt.show()
    return idxarray


def computeMeans(idxarray, X, K, centroids):
    print(centroids)
    for i in range(K):
        idxs = np.array(np.where(idxarray == i))
        centroids[i] = np.mean(X[idxs], axis=1)
    return centroids


matplotlib_axes_logger.setLevel('ERROR')
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
data = loadmat("./input/ex7data2.mat")
X = data['X']
centroids = initializeCentroids(X, 3)
for i in range(10):
    plotCentroids(centroids, 3, colors, markers)
    idxarray = closestCentroid(X, centroids, colors, 3, markers)
    centroids = computeMeans(idxarray, X, 3, centroids)
