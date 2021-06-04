import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as sci


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    reg = 0
    total_cost = -(1 / m) * np.sum(y.T.dot(np.log(probability(theta, X))) + (1 - y).T.dot(
        np.log(1 - probability(theta, X)))) + (reg / 2 / m) * np.sum(np.square(theta[1:]))
    return total_cost


def probability(theta, x):
    return sigmoid(x.dot(theta))


def gradient(theta, X, y):
    m = X.shape[0]
    return (1 / m) * (X.T.dot(sigmoid(X.dot(theta)) - y))


def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)


def mapFeature(x1, x2):
    degree = 6
    out = np.ones((x1.shape[0], sum(range(degree + 2))))
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, curr_column] = np.power(x1, i - j) * np.power(x2, j)
            curr_column += 1
    return out


def graph(data1, data2, theta):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U, V = np.meshgrid(u, v)
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    X_poly = mapFeature(U, V)
    Z = X_poly.dot(theta)
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    plt.figure(num='Microchip Test')
    plt.contour(U, V, Z, levels=[0], cmap="Greys_r")
    plt.plot(data1[:, [1]], data1[:, [2]], 'k+', label="Accepted")
    plt.plot(data2[:, [1]], data2[:, [2]], 'y.', label="Failed")
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title("Microchip Test")
    plt.grid(True)
    plt.show()


data = pd.read_csv("./input/ex2data2.txt", header=None,
                   names=['Microchip 1', 'Microchip 2', 'Passed'])
X1 = data['Microchip 1'].values.T
X2 = data['Microchip 2'].values.T
X = mapFeature(X1, X2)
data.insert(0, 'Ones', 1)
[m, cols] = X.shape
y = np.array(data.iloc[:, 3])
y = y[:, np.newaxis]
data = np.array(data)
Accepted = data[np.array(np.where(y == 1))[0]]
Failed = data[np.array(np.where(y == 0))[0]]
theta = np.zeros((cols, 1))
theta = sci.minimize(fun=cost_function, x0=theta, args=(X, y.flatten())).x
graph(Accepted, Failed, theta)
