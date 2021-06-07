import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def plotData(X):
    plt.plot(X[:, [0]], X[:, [1]], 'bx')
    plt.axis([0, 30, 0, 30])


def estimateGaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.sum((X - mu)**2, axis=0) / X.shape[0]
    return mu, sigma2


def plotDecisionBoundary(X, mu, sigma2):
    plotData(X)
    u = np.linspace(0, 35, 71)
    v = np.linspace(0, 35, 71)
    U, V = np.meshgrid(u, v)
    X = np.concatenate([np.ravel(U).reshape(5041, 1),
                        np.ravel(V).reshape(5041, 1)], axis=1)
    Z = multiVariateGaussian(X, mu, sigma2)
    Z = Z.reshape(71, 71)
    plt.contour(U, V, Z, levels=[10**i for i in range(-20, -2, 3)])
    plt.show()


def multiVariateGaussian(X, mu, sigma2):
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)
    p = (1 / (2 * np.pi**(X.shape[1] / 2) * np.linalg.det(sigma2)**0.5)) * \
        np.exp(-0.5 * np.sum((X - mu).dot(np.linalg.pinv(sigma2)) * (X - mu), axis=1))
    return p


data = loadmat("./input/ex8data1.mat")
X = data['X']
[mu, sigma2] = estimateGaussian(X)
p = multiVariateGaussian(X, mu, sigma2)
plotDecisionBoundary(X, mu, sigma2)
