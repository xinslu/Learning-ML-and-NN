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


data = pd.read_csv("./input/ex1data2.txt", header=None,
                   names=['Size', 'Bedrooms', 'Price'])
data = (data - data.mean()) / data.std()
data.insert(0, 'Ones', 1)
cols = data.shape[1]
m = data.shape[0]
X = np.array(data.iloc[:, 0:cols - 1])
Y = np.array(data.iloc[:, cols - 1:cols])
data = np.array(data)
theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
graph_prediction(X, theta, data)
