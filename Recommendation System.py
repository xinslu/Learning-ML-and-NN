import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def plotData(Y):
    fig, ax = plt.subplots()
    ax.imshow(Y, extent=[0, 1, 0, 1])
    plt.show()


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambdax):
    X = np.reshape(params[0: (num_movies * num_features)],
                   (num_movies, num_features))
    Theta = np.reshape(
        params[num_movies * num_features:], (num_users, num_features))
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    ratingerror = X.dot(Theta.T) - Y
    errorfactor = np.multiply(ratingerror, R)
    J = np.sum(errorfactor**2) / 2 + (lambdax / 2) * \
        np.sum(Theta**2) + (lambdax / 2) * np.sum(X**2)
    X_grad = errorfactor.dot(Theta) + lambdax * np.sum(X)
    Theta_grad = errorfactor.T.dot(X) + lambdax * np.sum(Theta)
    print(J)
    return J


def loadMovieList():
    file = open('./input/movie_ids.txt')
    movielist = np.zeros(len(file.readlines()))
    for i in file.readlines():
        movieraw = i.split()
        np.append(movielist, ' '.join(movieraw[1:]))
    return movielist


def assignRatings():
    my_ratings = np.zeros((1682, 1))
    my_ratings[1] = 4
    my_ratings[98] = 2
    my_ratings[7] = 3
    my_ratings[12] = 5
    my_ratings[54] = 4
    my_ratings[64] = 5
    my_ratings[66] = 3
    my_ratings[69] = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355] = 5
    return my_ratings


def normalizeRatings(Y, R):
    Ymean = np.zeros(Y.shape[0])
    Ynorm = np.zeros(Y.shape)
    for i in range(1, Y.shape[0]):
        idx = np.where(R[i] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        return Ymean, Ynorm


# data = loadmat("./input/ex8_movies.mat")
# Y = data['Y']
# data1 = loadmat("./input/ex8_movieparams.mat")
# X = data1['X']
# R = data['R']
# Theta = data1['Theta']
# num_users = 4
# num_movies = 5
# num_features = 3
# X = X[0: num_movies, 0: num_features]
# Theta = Theta[0: num_users, 0: num_features]
# Y = Y[0:num_movies, 0:num_users]
# R = R[0:num_movies, 0:num_users]
# params = np.append(X.ravel(), Theta.ravel(), axis=0)
# print(cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5))
movies = loadmat('./input/ex8_movies.mat')
my_ratings = assignRatings()
Y = movies['Y']
R = movies['R']
my_ratings = np.reshape(my_ratings, (my_ratings.shape[0], 1))
my_ratings0 = (my_ratings != 0).astype(int)
Y = np.append(Y, my_ratings, axis=1)
R = np.append(R, my_ratings0, axis=1)
[Ymean, Ynorm] = normalizeRatings(Y, R)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
lambdax = 1.5
X = np.random.normal(size=(num_movies, num_features))
Theta = np.random.normal(size=(num_users, num_features))
params = np.append(X.ravel(), Theta.ravel(), axis=0)
opts = {'maxiter': 100,
        'disp': True,
        'retall': True}
params = minimize(cofiCostFunc, params, method='BFGS', args=(Ynorm, R, num_users,
                                                             num_movies, num_features, lambdax))
X = np.reshape(params[0: (num_movies * num_features)],
               (num_movies, num_features))
Theta = np.reshape(
    params[num_movies * num_features:], (num_users, num_features))
p = X.dot(Theta.T)
