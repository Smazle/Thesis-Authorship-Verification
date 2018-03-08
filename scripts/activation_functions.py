#!/usr/bin/python3

import math
import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(x, np.zeros(x.shape))

def softmax(x):
    return np.exp(x) / float(sum(np.exp(x)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softsign(x):
    return x / (1 + np.abs(x))

def identity(x):
    return x

upper = 7.0
lower = -upper

x = np.arange(lower, upper, 0.01)

plt.plot(x, relu(x), label='Rectified Linear Unit')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, softsign(x), label='Softsign')
plt.plot(x, identity(x), label='Identity')
plt.ylim(-1.0, 1.0)
plt.xlim(lower, upper)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('h(x)')
plt.legend()
plt.show()
