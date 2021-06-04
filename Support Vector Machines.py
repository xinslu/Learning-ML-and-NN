import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.svm import SVC


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

def plotDecisionBoundary(X,y,svclassifier):
    plot_data(X,y)
    x_line = np.linspace(0, 4, 2)
    plt.ylim([1,5])
    y_line = (-1 * svclassifier.intercept_ - svclassifier.coef_[0][0] * x_line) / svclassifier.coef_[0][1]
    plt.plot(x_line, y_line)
    plt.show()


def plot_data(X,y):
    Accepted = X[np.array(np.where(y == 1))[0]]
    Failed = X[np.array(np.where(y == 0))[0]]
    plt.plot(Accepted[:, [0]], Accepted[:, [1]], 'k+')
    plt.plot(Failed[:, [0]], Failed[:, [1]], 'y.')

def accuracy(y_pred,testData):
    y_pred=np.reshape(y_pred,(y_pred.size,1))
    acc = np.mean(y_pred == testData[:,[2]])
    return acc*100

data = scipy.io.loadmat("./input/ex6data1.mat")
X = data['X']
y = data['y']
data=np.hstack((X, y))
totalData=X.shape[0]
[trainData, testData] = split_data(data, totalData)
svclassifier = SVC(kernel='linear',C=50)
svclassifier.fit(trainData[:,0:2], trainData[:,[2]].ravel())
y_pred = svclassifier.predict(testData[:,0:2])

