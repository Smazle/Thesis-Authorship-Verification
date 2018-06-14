#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

parser = argparse.ArgumentParser(
    'Produces a graph of the results of the prediction system')
parser.add_argument(
    '--image-out',
    help='Where to save the graph showing accuracies and errors.'
)
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
f, axarr = plt.subplots(2, sharex=True)
for weight in np.unique(weights):
    accs = accuracies[weights == weight]
    errs = accusation_errors[weights == weight]
    thresholds = thetas[weights == weight]

    axarr[0].plot(thresholds, accs, label=weight)
    axarr[1].plot(thresholds, errs, label=weight)

axarr[0].set_ylabel('Accuracy')
axarr[0].grid(True)

axarr[1].set_ylabel('Accusation Error')
axarr[1].grid(True)
axarr[1].legend()

axarr[1].set_xlabel('θ (Threshold)')
lgd = plt.legend(bbox_to_anchor=(1.25, 1), loc=7, fancybox=True)

if args.image_out is None:
    plt.show()
else:
    f.savefig(
        args.image_out,
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight'
    )

# Find the best configuration for each weight.
print('weight,allowed_error,theta,accuracy,accusation_error,tps,tns,fps,fns')
for weight in np.unique(weights):
    for allowed_error in np.linspace(0.1, 0.9, num=9):
        accs = accuracies[weights == weight]
        errs = accusation_errors[weights == weight]

        best_index = np.argmax(accs * (errs < allowed_error))

        tp = tps[weights == weight][best_index]
        tn = tns[weights == weight][best_index]
        fp = fps[weights == weight][best_index]
        fn = fns[weights == weight][best_index]
        theta = thetas[weights == weight][best_index]

        print('{},{},{},{},{},{},{},{},{}'
              .format(weight, allowed_error, theta, accs[best_index],
                      errs[best_index], tp, tn, fp, fn))
