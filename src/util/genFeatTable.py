#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse


parse = argparse.ArgumentParser(
    'Takes a list of feature names, and greedyly selected\
                features, and prints out a table'
)


parse.add_argument(
    'names',
    type=str,
    help='Path to file containing feature names'
)

parse.add_argument(
    'features',
    type=str,
    help='Path to file containing feature selected features'
)

parse.add_argument(
    '--limit',
    type=int,
    help='Imposes a limit on how many features to look at'
)


base = ['\\begin{table}', '\\centering', '\\begin{tabular}{ccc}',
        'Feature Type & Frequency Rank & Feature \\\\ \hline']

args = parse.parse_args()

features = np.loadtxt(args.features, delimiter=',')
features = features[:np.argmax(features, axis=0)[-1]]

names = np.loadtxt(args.names, delimiter='\n', dtype=str)

for idx, _ in features:
    feat = names[int(idx)]
    name, feat = feat.split('\t')

    name = name.split('-')

    rank = name[-1]
    name = name[:-1]
    name[0] = name[0][0].upper() + name[0][1:]
    name.append('Gram')
    name = ' '.join(name)

    base.append(' & '.join([name, rank, feat]) + '\\\\')

base.append('\\end{tabular}')
base.append('\\end{table}')

print(len(features))
print('\n'.join(base))
