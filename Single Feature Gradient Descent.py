import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graph_prediction(X, theta, data):
    x = np.linspace(X.min(), X.max(), 1000)
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
X = np.array(data.iloc[:, 0:cols - 1])
Y = np.array(data.iloc[:, cols - 1:cols])
theta = np.zeros((2, 1))
data = np.array(data)
iterations = 1500
alpha = 0.01
hypothesis = X.dot(theta)
cost = np.zeros(iterations)
for i in range(iterations):
    prediction = X.dot(theta)
    theta = theta - (X.T.dot((prediction - Y)) / m) * alpha
    cost[i] = np.sum((prediction - Y)**2) / 2 / m

graph_cost(iterations, cost)
