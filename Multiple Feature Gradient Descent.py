import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graph_prediction(X, theta, data):
    x = np.linspace(X.min(), X.max(), 1000)
    plt.plot(x, theta[0] + x * theta[1], ':r')
    plt.plot(data[:, [1]], data[:, [3]], 'rx')
    plt.xlabel("Size")
    plt.ylabel("Price")
    plt.title('Size vs. Price')
    plt.legend(['Prediction', 'Data'])
    plt.xlim([-1, 4])
    plt.grid(True)
    plt.show()


def graph_cost(iterations, cost):
    plt.plot(np.arange(iterations), cost, 'r.')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')
    plt.show()


data = pd.read_csv("./input/ex1data2.txt", header=None,
                   names=['Size', 'Bedrooms', 'Price'])
data = (data - data.mean()) / data.std()
data.insert(0, 'Ones', 1)
cols = data.shape[1]
m = data.shape[0]
X = np.array(data.iloc[:, 0:cols - 1])
Y = np.array(data.iloc[:, cols - 1:cols])
data = np.array(data)
theta = np.zeros((3, 1))
iterations = 1500
alpha = 0.01
hypothesis = X.dot(theta)
cost = np.zeros(iterations)
for i in range(iterations):
    prediction = X.dot(theta)
    theta = theta - (X.T.dot((prediction - Y)) / m) * alpha
    cost[i] = np.sum((prediction - Y)**2) / 2 / m

graph_cost(iterations, cost)
