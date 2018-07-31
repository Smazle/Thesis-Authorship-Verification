#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


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

args = parser.parse_args()

data = pd.read_csv(sys.stdin)

weights = data.as_matrix(columns=['weight'])
thetas = data.as_matrix(columns=['threshold'])
accuracies = data.as_matrix(columns=['accuracy'])
accusation_errors = data.as_matrix(columns=['accusation_error'])
tps = data.as_matrix(columns=['tps'])
tns = data.as_matrix(columns=['tns'])
fps = data.as_matrix(columns=['fps'])
fns = data.as_matrix(columns=['fns'])

# Generate graph.
f, axarr = plt.subplots(2, 3)
f.tight_layout()
ax = f.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor='none', top='off', bottom='off', left='off', right='off')

for weight in np.sort(np.unique(weights)):
    # Different weight configurations
    # if weight not in [
    #        'Exp λ=0.25', 'Exp λ=0.5', 'Exp λ=0.75', 'Exp λ=1.0', 'Exp λ=0.0'
    # ]:
    # if weight not in [
    #        'Exp λ=0.0', "Text Length"
    # ]:
    # if weight not in [
    #        'Exp λ=0.0', "Exp λ=0.25 + Text Length"
    # ]:
    if weight not in ['Exp λ=0.0', 'Majority Vote']:
        continue
    weight_name = get_weight_report_name(weight)

    # Between 0 and 0.2.
    accs = accuracies[np.logical_and(weights == weight, thetas <= 0.2)]
    errs = accusation_errors[np.logical_and(weights == weight, thetas <= 0.2)]
    thresholds = thetas[np.logical_and(weights == weight, thetas <= 0.2)]

    axarr[0, 0].plot(thresholds, accs, label=weight_name)
    axarr[1, 0].plot(thresholds, errs, label=weight_name)

    # Between 0.4 and 0.6.
    accs = accuracies[np.logical_and(
        weights == weight, np.logical_and(thetas >= 0.4, thetas <= 0.6))]
    errs = accusation_errors[np.logical_and(
        weights == weight, np.logical_and(thetas >= 0.4, thetas <= 0.6))]
    thresholds = thetas[np.logical_and(
        weights == weight, np.logical_and(thetas >= 0.4, thetas <= 0.6))]

    axarr[0, 1].plot(thresholds, accs, label=weight_name)
    axarr[1, 1].plot(thresholds, errs, label=weight_name)

    # Between 0.8 and 1.0.
    accs = accuracies[np.logical_and(weights == weight, thetas >= 0.8)]
    errs = accusation_errors[np.logical_and(weights == weight, thetas >= 0.8)]
    thresholds = thetas[np.logical_and(weights == weight, thetas >= 0.8)]

    axarr[0, 2].plot(thresholds, accs, label=weight_name)
    axarr[1, 2].plot(thresholds, errs, label=weight_name)

axarr[0, 0].grid(True)
axarr[1, 0].grid(True)
axarr[0, 1].grid(True)
axarr[1, 1].grid(True)
axarr[0, 2].grid(True)
axarr[1, 2].grid(True)

ax.set_xlabel('θ (Threshold)')
# ax.set_ylabel('Accuracy')
axarr[0, 0].set_ylabel('Accuracy')
axarr[1, 0].set_ylabel('Accusation Error')

lgd = axarr[0, 0].legend(bbox_to_anchor=(0.2, -1.5), loc='upper left', ncol=5)

if args.image_out is None:
    plt.show()
else:
    f.savefig(
        args.image_out,
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight',
        format='pdf')
