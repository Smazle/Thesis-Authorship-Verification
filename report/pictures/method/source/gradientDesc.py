# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# The data to fit
m = 20
theta0_true = 2
theta1_true = 0.5
x = np.linspace(-1, 1, m)
y = theta0_true + theta1_true * x


def cost_func(theta0, theta1):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((y - hypothesis(x, theta0, theta1))**2, axis=2) / 2


def hypothesis(x, theta0, theta1):
    """Our "hypothesis function", a straight line."""
    return theta0 + theta1 * x


# First construct a grid of (theta0, theta1) parameter pairs and their
# corresponding cost function values.
theta0_grid = np.linspace(-1, 4, 101)
theta1_grid = np.linspace(-5, 5, 101)
J_grid = cost_func(theta0_grid[:, np.newaxis, np.newaxis],
                   theta1_grid[np.newaxis, :, np.newaxis])

# A labeled contour plot for the RHS cost function
X, Y = np.meshgrid(theta0_grid, theta1_grid)
contours = plt.contour(X, Y, J_grid, 30)
plt.clabel(contours)
# The target parameter values indicated on the cost function contour plot
plt.scatter([theta0_true] * 2, [theta1_true] * 2, s=[50, 10], color=['k', 'w'])

N = 5
alpha = 0.7
theta = [np.array((0, 0))]
J = [cost_func(*theta[0])[0]]
for j in range(N - 1):
    last_theta = theta[-1]
    this_theta = np.empty((2,))
    this_theta[0] = last_theta[0] - alpha / m * np.sum(
        (hypothesis(x, *last_theta) - y))
    this_theta[1] = last_theta[1] - alpha / m * np.sum(
        (hypothesis(x, *last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_func(*this_theta))


# Annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# Also plot the fit function on the LHS data plot in a matching colour.
colors = ['b', 'g', 'm', 'c', 'orange']
for j in range(1, N):
    plt.annotate('', xy=theta[j], xytext=theta[j - 1],
                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                 va='center', ha='center')
plt.scatter(*zip(*theta), c=colors, s=40, lw=0)

plt.savefig('GradientDesc.png')
