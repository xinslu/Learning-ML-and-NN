import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from decimal import Decimal


def displayData(X):
    fig, axis = plt.subplots(10, 10)
    for i, ax in enumerate(axis.flat):
        ax.imshow(X[i].reshape(20, 20), cmap='gray')
        ax.set_aspect("auto")
        ax.axis('off')
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_reg):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')
    m = len(X)
    X = np.column_stack((np.ones((m, 1)), X))
    z2 = X.dot(Theta1.T)
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    eye_matrix = np.eye(num_labels)
    y = eye_matrix[(y - 1).ravel()]
    J = (1.0 / m) * np.sum(-np.multiply(y, np.log(a3)) -
                           np.multiply((1 - y), np.log(1 - a3)))
    reg = (lambda_reg / 2 / m) * \
        (np.sum(np.sum(Theta1[:, 1:]**2)) + np.sum(np.sum(Theta2[:, 1:]**2)))
    J += reg
    d3 = a3 - y
    d2 = np.multiply(d3.dot(Theta2[:, 1:]), sigmoidGradient(z2))
    delta1 = d2.T.dot(X)
    delta2 = d3.T.dot(a2)
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1reg = Theta1 * (lambda_reg / m)
    Theta2reg = Theta2 * (lambda_reg / m)
    Theta1_grad = Theta1_grad + Theta1reg
    Theta2_grad = Theta2_grad + Theta2reg
    Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
    Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]
    grad = np.concatenate((Theta1_grad.reshape(
        Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
    return J


def gradient(nn_params, input_layer_size, hidden_layer_size,
             num_labels, X, y, lambda_reg):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')
    m = len(X)
    X = np.column_stack((np.ones((m, 1)), X))
    z2 = X.dot(Theta1.T)
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    eye_matrix = np.eye(num_labels)
    y = eye_matrix[(y - 1).ravel()]
    d3 = a3 - y
    d2 = np.multiply(d3.dot(Theta2[:, 1:]), sigmoidGradient(z2))
    delta1 = d2.T.dot(X)
    delta2 = d3.T.dot(a2)
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1reg = Theta1 * (lambda_reg / m)
    Theta2reg = Theta2 * (lambda_reg / m)
    Theta1_grad = Theta1_grad + Theta1reg
    Theta2_grad = Theta2_grad + Theta2reg
    Theta1_grad[:, 0] = Theta1_grad_unregularized[:, 0]
    Theta2_grad[:, 0] = Theta2_grad_unregularized[:, 0]
    grad = np.concatenate((Theta1_grad.reshape(
        Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
    return grad


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 0.0001
    for p in range(theta.shape[0] * theta.shape[1]):
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad


def checkNNGradients(lambda_reg=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X = debugInitializeWeights(m, input_layer_size - 1)
    nn_params = np.append(
        Theta1.ravel(), Theta2.ravel(), axis=0)
    nn_params = nn_params.reshape(nn_params.shape[0], 1)

    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size,
                              num_labels, X, y, lambda_reg)
    y = 1 + np.mod(range(m), num_labels).T
    [cost, grad] = nnCostFunction(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdax)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    print(np.c_[numgrad, grad])
    print(Decimal(np.linalg.norm(grad.flatten() - numgrad.flatten())) /
          Decimal(np.linalg.norm(grad.flatten() + numgrad.flatten())))


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10
    return W


def learning(initial_nn_params):
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_reg = 1

    def gradientlearning(p):
        return gradient(p, input_layer_size, hidden_layer_size,
                        num_labels, X, y, lambda_reg)

    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size,
                              num_labels, X, y, lambda_reg)
    options = {'disp': True, 'maxiter': 200}
    nn_params = minimize(costFunc, jac=gradientlearning, x0=initial_nn_params,
                         method='CG', options=options)
    return nn_params.x


data = loadmat("./input/ex4data1.mat")
thetas = loadmat('./input/ex4weights.mat')
Theta1 = thetas['Theta1']
Theta2 = thetas['Theta2']
X = data['X']
y = data['y']
nn_params = np.append(Theta1.ravel(), Theta2.ravel(), axis=0)
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(
    initial_Theta1.ravel(), initial_Theta2.ravel(), axis=0)
nn_params = learning(initial_nn_params)
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, input_layer_size + 1), order='F')
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, hidden_layer_size + 1), order='F')
