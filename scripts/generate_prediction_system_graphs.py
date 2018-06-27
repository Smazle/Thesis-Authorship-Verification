#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

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
f = plt.figure(figsize=(20, 10))
ax1 = f.add_subplot(211)
ax2 = f.add_subplot(212, sharex=ax1)

for weight in np.sort(np.unique(weights)):
    accs = accuracies[weights == weight]
    errs = accusation_errors[weights == weight]
    thresholds = thetas[weights == weight]

    weight = weight.replace('+ Text Length', '+ TL')
    ax1.plot(thresholds, accs, label=weight)
    ax2.plot(thresholds, errs, label=weight)

ax1.set_ylabel('Accuracy', fontsize=15)
ax1.grid(True)

ax2.set_ylabel('Accusation Error', fontsize=15)
ax2.grid(True)
ax2.legend()

ax2.set_xlabel('θ (Threshold)', fontsize=15)
lgd = plt.legend(
    bbox_to_anchor=(0.11, -0.2), loc='upper left', prop={'size': 15}, ncol=5)

if args.image_out is None:
    plt.show()
else:
    f.savefig(
        args.image_out,
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight',
        dpi=128)

# Find the best configuration for each weight.
print('{:^15}{:^13}{:^10}{:^10}{:^18}{:^8}{:^8}{:^8}{:^8}'.format(
    'weight', 'allowed_error', 'theta', 'accuracy', 'accusation_error', 'tps',
    'tns', 'fps', 'fns'))
for weight in np.sort(np.unique(weights)):
    for allowed_error in np.linspace(0.1, 0.9, num=9):
        accs = accuracies[weights == weight]
        errs = accusation_errors[weights == weight]

        best_index = np.argmax(accs * (errs < allowed_error))

        tp = tps[weights == weight][best_index]
        tn = tns[weights == weight][best_index]
        fp = fps[weights == weight][best_index]
        fn = fns[weights == weight][best_index]
        theta = thetas[weights == weight][best_index]

        p_weight = weight.replace('+ Text Length', '+')
        print('{:^15}{:^13.1f}{:^10.3f}{:^10.5f}{:^18.3f}{:^8}{:^8}{:^8}{:^8}'
              .format(p_weight, allowed_error, theta, accs[best_index],
                      errs[best_index], tp, tn, fp, fn))

    print()
