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
    plt.contour(U, V, Z, levels=[10**i for i in range(-20, -1, 3)])
    plt.show()


def multiVariateGaussian(X, mu, sigma2):
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)
    p = (1 / (2 * np.pi**(X.shape[1] / 2) * np.linalg.det(sigma2)**0.5)) * \
        np.exp(-0.5 * np.sum((X - mu).dot(np.linalg.pinv(sigma2)) * (X - mu), axis=1))
    return p


def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        cvPredictions = (pval < epsilon)
        fp = np.sum(np.logical_and(cvPredictions == 1, yval == 0))
        tp = np.sum((np.logical_and(cvPredictions == 1, yval == 1)))
        fn = np.sum(np.logical_and(cvPredictions == 0, yval == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


data = loadmat("./input/ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']
[mu, sigma2] = estimateGaussian(X)
p = multiVariateGaussian(X, mu, sigma2)
pval = multiVariateGaussian(Xval, mu, sigma2)
pval = np.reshape(pval, (pval.shape[0], 1))
[epsilon, F1] = selectThreshold(yval, pval)
outliers = np.where(p < epsilon)
plt.plot(X[outliers][:, [0]], X[outliers][:, [1]], 'ro', markersize=12)
plotDecisionBoundary(X, mu, sigma2)
