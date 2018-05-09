#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def exp(xs, lamb=1.0):
    return np.exp(-lamb * xs)


def exp_norm(xs, lamb=1.0):
    xs = exp(xs, lamb)
    return xs / np.sum(xs)


xs = np.arange(0, 36, 1)

for i in np.linspace(0, 1.0, 20):
    print(i)
    plt.plot(xs, exp(xs, lamb=i))
plt.show()

for i in np.linspace(0, 1.0, 20):
    print(i)
    plt.plot(xs, exp_norm(xs, lamb=i))
plt.show()
