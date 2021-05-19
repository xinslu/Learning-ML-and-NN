import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
fin=open("./input/ex1data1.txt","r")
a=[]
lines=[]
for i in fin.readlines():
	a=[float(j)for j in i.strip("\n").split(",")]
	lines.append(a)
data = np.array(lines)
m=data.shape[0]
theta = np.zeros((2, 1))
X = np.insert(data[:,[0]],0,np.ones((m,),dtype=int), axis=1)
Y=data[:,[1]]
iterations=1500
alpha = 0.01
hypothesis=X.dot(theta)
for i in range(iterations):
	prediction=X.dot(theta)
	theta=theta-(X.T.dot((prediction-Y))/m)*alpha
print(theta)
x = np.linspace(X.min(),X.max(), 1000)
plt.plot(x, theta[0]+x*theta[1], ':r')
lineX=np.linspace(X.min(), X.max(), 100)
plt.xlim([5,25])
plt.plot(data[:,[0]],data[:,[1]],'rx')
plt.grid(True)
plt.show()