import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC


def plot_decision_regions(X, y, classifier, resolution=0.01):
    plot_data(X, y)
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, levels=[0.15])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def plot_data(X, y):
    Accepted = X[np.array(np.where(y == 1))[0]]
    Failed = X[np.array(np.where(y == 0))[0]]
    plt.plot(Accepted[:, [0]], Accepted[:, [1]], 'k+')
    plt.plot(Failed[:, [0]], Failed[:, [1]], 'y.')


def split_data(data, totalData):
    trainIndex, testIndex = list(), list()
    for i in range(1, totalData):
        if np.random.uniform(0, 1) < 0.7:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = data[trainIndex]
    testData = data[testIndex]
    return trainData, testData


data = loadmat("./input/ex6data2.mat")
X = data['X']
y = data['y']
data = np.hstack((X, y))
totalData = X.shape[0]
[trainData, testData] = split_data(data, totalData)
svclassifier = SVC(kernel='rbf', C=1, gamma=50)
svclassifier.fit(trainData[:, 0:2], trainData[:, [2]].ravel())
plot_decision_regions(X, y, svclassifier)
