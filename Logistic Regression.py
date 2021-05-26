import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as sci

def sigmoid(z):
	return 1/(1+np.exp(-z))

def cost_function(theta, X, y):
    total_cost = -(1/m)*np.sum(y*np.log(probability(theta, X)) + (1-y)*np.log(1-probability(theta,X)))
    return total_cost

def probability(theta, x):
    return sigmoid(x.dot(theta))

def gradient(theta, X, y):
    m = X.shape[0]
    return (1/m)*(X.T.dot(sigmoid(X.dot(theta))-y))

def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)



def graph_cost(iterations,cost):
	plt.plot(np.arange(iterations),cost,'r.')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Error vs. Training Epoch')
	plt.show()


data = pd.read_csv("./input/ex2data1.txt" , header=None, names=['Exam 1', 'Exam 2','Admitted'])
data.insert(0, 'Ones', 1)
[m,cols]=data.shape
X = np.array(data.iloc[:,0:cols-1])
y = np.array(data.iloc[:,cols-1:cols])
y = y[:, np.newaxis]
data=np.array(data)
Data1=data[np.array(np.where(y == 1))[0]]
Data2=data[np.array(np.where(y == 0))[0]]
iterations= 1000
alpha=0.001
theta=np.zeros((3, 1))
cost = np.zeros(iterations)
theta = sci.fmin_tnc(func = cost_function, x0 = theta.flatten(), fprime = gradient, disp=False, args = (X, y.flatten()))[0]
# accuracy(X, y.flatten(), theta, 0.5) has an accuracy of 89%
plt.figure(num='Exam Scores')
plt.plot(Data1[:,[1]],Data1[:,[2]],'k+')
plt.plot(Data2[:,[1]],Data2[:,[2]],'y.')
x = np.linspace(X.min(),X.max(), 1000)
plt.xlim([25,105])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.plot(x, (theta[0]+x*theta[1])/(-theta[2]), ':r')
plt.title("Admitted vs Not Admitted")
plt.grid(True)
plt.show()

