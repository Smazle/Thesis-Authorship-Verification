#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


parse = argparse.ArgumentParser(
    'Produces a graph of the results of the prediction system'
)

parse.add_argument(
    'data',
    help='Path to the data needed presenting',
    type=str
)

args = parse.parse_args()

data = pd.read_csv(args.data, sep=r'\s+')
print(data)

thetas = data.as_matrix(columns=['Theta'])
weights = data.as_matrix(columns=['Weights'])
accuracies = data.as_matrix(columns=['ACC'])
errors = data.as_matrix(columns=['ERR'])

data = data.as_matrix(columns=['Theta', 'Weights', 'ACC', 'ERR'])

for weight in np.unique(weights):
    label = 'Weight {}'.format(weight)
    thetas_w = thetas[weights == weight]
    accuracies_w = accuracies[weights == weight]

    plt.plot(thetas_w, accuracies_w, label=label)
plt.xlabel('Threshold (Theta)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('Accuracy.png')
plt.show()
plt.clf()

for weight in np.unique(weights):
    label = 'Weight {}'.format(weight)
    thetas_w = thetas[weights == weight]
    errors_t = errors[weights == weight]

    plt.plot(thetas_w, errors_t, label=label)
plt.xlabel('Threshold (Theta)')
plt.ylabel('Accusation Error')
plt.grid(True)
plt.legend()
plt.savefig('Accusation_Error.png')
plt.show()
plt.clf()

valid = data[data[:, 3] < 0.1]
best_index = np.argmax(valid[:, 2])
best_conf = valid[best_index]
print('The best legal configuration are theta={}, weight={},\
        accuracy={}, error={}'
      .format(best_conf[0], best_conf[1], best_conf[2], best_conf[3]))
