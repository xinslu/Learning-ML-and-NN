import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def computeCost(theta, X, y):
    theta = theta.reshape(X.shape[1], 1)
    reg = 0.1
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    h = h.reshape(h.shape[0], 1)
    total_cost = -(1 / m) * np.sum(y.T.dot(np.log(h)) + (1 - y).T.dot(
        np.log(1 - h))) + (reg / 2 / m) * np.sum(np.square(theta[1:]))
    return total_cost


def computeGradient(theta, X, y):
    theta = theta.reshape(X.shape[1], 1)
    reg = 0.1
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    h = h.reshape(h.shape[0], 1)
    theta[0] = 0
    grad = (1 / m) * (X.T.dot(h - y)) + (reg / m) * (theta)
    return grad.flatten()


def displayData(X):
    fig, axis = plt.subplots(10, 10)
    for i, ax in enumerate(axis.flat):
        ax.imshow(X.reshape((20, 20)), cmap='gray')
        ax.set_aspect("auto")
        ax.axis('off')
    fig.subplots_adjust(wspace=0.0001, hspace=0.0001)
    plt.show()


def multiClassClassification(X, y):
    n = X.shape[1]
    all_theta = np.zeros((10, n))
    for c in range(1, 11):
        initial_theta = np.zeros((n))
        options = {'disp': True}
        theta = minimize(computeCost, jac=computeGradient, x0=initial_theta,
                         args=(X, y == c), method='CG', options=options)
        print("THeta", theta.x)
        all_theta[c - 1] = theta.x
    return all_theta


def predictOneVsAll(all_theta, X):
    prediction = sigmoid(X.dot(all_theta.T))
    return np.argmax(prediction, axis=1)


data = loadmat("./input/ex3data1.mat")
X = data['X']
y = data['y']
selected = X[np.random.permutation(X.shape[0])][:100]
X = np.insert(X, 0, 1, axis=1)
theta_t = np.zeros((X.shape[1]))
all_theta = multiClassClassification(X, y)
print(all_theta)
pred = predictOneVsAll(all_theta, X)
print(pred)
pred.reshape((pred.shape[0], 1))
print(np.mean(pred == y))
