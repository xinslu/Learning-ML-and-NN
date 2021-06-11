import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graph_prediction(X, theta, data):
    x = np.linspace(X.min(), X.max(), 1000)
    print(x.shape, theta.shape)
    plt.plot(x, theta[0] + x * theta[1], '.r')
    plt.xlim([5, 25])
    plt.plot(data[:, [1]], data[:, [2]], 'rx')
    plt.grid(True)
    plt.show()


def graph_cost(iterations, cost):
    plt.plot(np.arange(iterations), cost, 'r.')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')
    plt.show()


data = pd.read_csv("./input/ex1data1.txt", header=None,
                   names=['Size', 'Profit'])
data.insert(0, 'Ones', 1)
cols = data.shape[1]
m = data.shape[0]
data = np.array(data)
X = data[:, :cols - 1]
Y = data[:, cols - 1:]
theta = np.zeros((2, 1))
iterations = 1000
alpha = 0.01
cost = np.zeros(m)
for j in range(m):
    idx = np.random.randint(m)
    prediction = theta[0] + theta[1] * X[idx]
    theta = theta - (prediction - Y[idx]).dot(X[idx]) * alpha
    cost[j] = np.sum((prediction - Y[idx])**2) / 2
graph_prediction(X, theta, data)
graph_cost(m, cost)
