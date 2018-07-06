#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc


def get_weight_report_name(previous_name):
    if previous_name == 'Exp λ=0.0':
        return r'$P_{u}$'
    elif previous_name == 'Exp λ=0.25':
        return r'$P_{exp_{0.25}}$'
    elif previous_name == 'Exp λ=0.5':
        return r'$P_{exp_{0.50}}$'
    elif previous_name == 'Exp λ=0.75':
        return r'$P_{exp_{0.75}}$'
    elif previous_name == 'Exp λ=1.0':
        return r'$P_{exp_{1.00}}$'
    elif previous_name == 'Maximum':
        return r'$P_{max}$'
    elif previous_name == 'Minimum':
        return r'$P_{min}$'
    elif previous_name == 'Majority Vote':
        return r'$P_{MV}$'
    elif previous_name == 'Text Length':
        return r'$P_l$'
    elif previous_name == 'Exp λ=0.25 + Text Length':
        return r'$P_{lexp_{0.25}}$'
    else:
        raise Exception('Unknown weight function')


parser = argparse.ArgumentParser(
    'Produces a graph of the results of the prediction system.' +
    'The output is given as stdin to the program.')
parser.add_argument(
    '--image-out',
    help='Where to save the graph showing accuracies and errors.')

parser.add_argument(
    'smalldata',
    help='96/04 split data file, produed by pred system',
    type=str)

parser.add_argument(
    'bigdata', help='50/50 split data file, produed by pred system', type=str)

args = parser.parse_args()

# 96/04 data
small_data = pd.read_csv(args.smalldata)

# 50/50 data
big_data = pd.read_csv(args.bigdata)

data = small_data

weights = data.as_matrix(columns=['weight'])

# Generate graph.
for weight in np.sort(np.unique(weights)):
    roc = plt.figure(2)
    ax = plt.subplot(111)
    plots_X = []
    plots_Y = []

    for i in range(2):
        accuracies = data.as_matrix(columns=['accuracy'])
        tps = data.as_matrix(columns=['tps'])
        tns = data.as_matrix(columns=['tns'])
        fps = data.as_matrix(columns=['fps'])
        fns = data.as_matrix(columns=['fns'])

        tps = tps[weights == weight]
        tns = tns[weights == weight]
        fps = fps[weights == weight]
        fns = fns[weights == weight]

        tpr = np.array([x / (x + fn) for x, fn in zip(tps, fns)])
        fpr = np.array([x / (x + tn) for x, tn in zip(fps, tns)])

        s = np.argsort(fpr)
        tpr = tpr[s]
        fpr = fpr[s]

        tpr, fpr = zip(*np.unique(list(zip(tpr, fpr)), axis=0))

        plots_X.append(fpr)
        plots_Y.append(tpr)

        data = big_data

    plot_1, = ax.plot(
        plots_X[0],
        plots_Y[0],
        label='96/04 AUC {}'.format(auc(plots_X[0], plots_Y[0])))
    plot_2, = ax.plot(
        plots_X[1],
        plots_Y[1],
        label='50/50 AUC {}'.format(auc(plots_X[1], plots_Y[1])))

    ax.plot([0.0, 1.0], [0.0, 1.0], '--')
    ax.set_title(weight)
    ax.legend(handles=[plot_1, plot_2])
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.show()
    roc.clf()
    data = small_data
