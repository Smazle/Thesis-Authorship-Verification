#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.neighbors import KNeighborsClassifier


parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training'
)

parser.add_argument(
    'validationFile',
    type=str,
    help='Path to file containing raw features'
)

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection'
)

parser.add_argument(
    '--K',
    type=int,
    help='The parameter selected value of K'
)

parser.add_argument(
    '--p',
    type=int,
    help='The parameter selected value of p'
)

args = parser.parse_args()

classifier = KNeighborsClassifier(n_neighbors=args.K, p=args.p)

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

fs = FeatureSearch(None, None, None)
fs.__generateData__(args.validationFile)

output = fs.__evaluate_classifier__(classifier, features)

print('Validation Result, with K={} and p={}:\
        {}'.format(args.K, args.p, output))
