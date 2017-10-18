import numpy as np
import matplotlib.pyplot as plt

# load data
data_x = np.loadtxt('logistic_x.txt')
data_y = np.loadtxt('logistic_y.txt')

data_x_origin = data_x
# add intercept term
data_x = np.hstack((data_x, np.ones((data_x.shape[0], 1))))

# define loss function


def J(theta):
    return np.sum(np.log(np.exp(- data_x.dot(theta) * data_y) + 1)) / data_x.shape[0]

# define function g(z)


def g(theta):
    return 1 / (np.exp(- data_x.dot(theta) * data_y) + 1)

# define gradient of loss function


def dJ(theta):
    return - np.sum(np.vstack((1 - g(theta)) * data_y) * data_x, axis=0) / data_x.shape[0]

# define Hessian matrix of loss function


def H(theta):
    return data_x.transpose().dot(np.vstack(g(theta) * (1 - g(theta)) * data_y * data_y) * data_x) / data_x.shape[0]

# use Newton's method to minimize loss function, repeat iteration until
# the change of theta is small enough
error = 1e-8
theta = np.array([0, 0, 0])
iter = 0
while(True):
    d_theta = np.linalg.inv(H(theta)).dot(dJ(theta))

    if np.sqrt(np.sum(d_theta * d_theta)) < error:
        break

    theta = theta - d_theta
    print("iter" + str(iter) + ": ")
    print("theta = ", theta)
    print("loss = ", J(theta))
    iter += 1

mask_T = data_y == 1
mask_F = data_y == -1

# plot data points('x' for y = 1; '+' for y = -1)
plt.plot(data_x_origin[mask_T][:, 0], data_x_origin[mask_T][:, 1], 'rx')
plt.plot(data_x_origin[mask_F][:, 0], data_x_origin[mask_F][:, 1], 'b+')

# plot the decision boundary
k = - theta[0] / theta[1]
b = -theta[2] / theta[1]
plt.plot([0, 8], [b, 8 * k + b], 'g')

plt.show()
