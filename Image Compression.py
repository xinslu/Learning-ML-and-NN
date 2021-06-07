import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import scipy.spatial


def initializeCentroids(X, K):
    randomidx = np.random.permutation(X.shape[0])
    centroids = X[randomidx[0:K]]
    return centroids


def plotCentroids(centroids, K):
    for i in range(K):
        plt.scatter(centroids[:, [0]][i], centroids[:, [1]]
                    [i], s=100)
    plt.show()


def plotPoints(X):
    plt.scatter(X[:, [0]], X[:, [1]], s=3)


def closestCentroid(X, centroids, K):
    D = scipy.spatial.distance.cdist(X, centroids, 'euclidean')
    return np.argmin(D.T, axis=0)


def computeMeans(idxarray, X, K, centroids):
    for i in range(K):
        idxs = np.array(np.where(idxarray == i))
        centroids[i] = np.mean(X[idxs], axis=1)
    return centroids


matplotlib_axes_logger.setLevel('ERROR')
img = mpimg.imread('./input/bird_small.png')
plt.imshow(img)
plt.show()
X = img.reshape(img.shape[0] * img.shape[1], 3)
centroids = initializeCentroids(X, 16)
for i in range(10):
    idxarray = closestCentroid(X, centroids, 16)
    centroids = computeMeans(idxarray, X, 16, centroids)
idxarray = closestCentroid(X, centroids, 16)
X_recovered = centroids[idxarray]
X_recovered = X_recovered.reshape(img.shape[0], img.shape[1], 3)
plt.imshow(X_recovered)
plt.show()
