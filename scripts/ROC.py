#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from matplotlib import colors as mcolors

colors = list(mcolors.BASE_COLORS.keys())


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
    'datafiles',
    nargs='+',
    type=str,
    help='Path to prediction system files, order of which is\
        determined by order argument')

parser.add_argument(
    '-o',
    '--order',
    nargs='+',
    type=str,
    default=['CNN3_50', 'CNN3_04', 'CNN6_50', 'CNN6_04', 'RNN_50', 'RNN_04'])

parser.add_argument(
    '-l',
    '--legend',
    nargs='+',
    type=str,
    default=[
        'char_CNN 50/50', 'char_CNN 96/04', 'char_word_CNN 50/50',
        'char_word_CNN 96/04', 'sent_RNN 50/50', 'sent_RNN 96/04'
    ])

args = parser.parse_args()

data_dict = {
    key: pd.read_csv(value)
    for key, value in zip(args.order, args.datafiles)
}

weights = data_dict[args.order[0]].as_matrix(columns=['weight'])

roc = plt.figure(2, figsize=(10, 8))
ax = plt.subplot(111)

ax.plot([0.0, 1.0], [0.0, 1.0], '--')
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

plots = []

for idx, key in enumerate(args.order):
    plots_X = 0
    plots_Y = 0

    prev = 0
    w = 0
    data = data_dict[key]
    for weight in np.sort(np.unique(weights)):

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

        new = auc(fpr, tpr)
        if new > prev:
            plots_X = fpr
            plots_Y = tpr
            w = get_weight_report_name(weight)
            prev = new

    if idx % 2 == 0:
        plot, = ax.plot(
            plots_X,
            plots_Y,
            c=colors[int(idx / 2)],
            label='{}, {}, AUC: {:^.3f}'.format(args.legend[idx], w,
                                                auc(plots_X, plots_Y)))
    else:
        plot, = ax.plot(
            plots_X,
            plots_Y,
            linestyle='dashed',
            c=colors[int(idx / 2)],
            label='{}, {}, AUC: {:^.3f}'.format(args.legend[idx], w,
                                                auc(plots_X, plots_Y)))

ax.legend(loc=4)
if args.image_out is not None:
    plt.savefig(args.image_out, format='pdf')
else:
    plt.show()
