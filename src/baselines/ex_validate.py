#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training')

parser.add_argument(
    'trainingFile',
    type=str,
    help='Path to training file containing raw features')

parser.add_argument(
    'validationFile', type=str, help='Path to file containing raw features')

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection')

parser.add_argument('--K', type=int, help='The parameter selected value of K')

parser.add_argument('--p', type=int, help='The parameter selected value of p')

args = parser.parse_args()

classifier = KNeighborsClassifier(n_neighbors=args.K, p=args.p)

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

train = FeatureSearch(None, None, None)
train.__generateData__(args.trainingFile)

validation = FeatureSearch(None, None, None)
validation.__generateData__(args.validationFile)
validation.data = train.scaler.transform(validation.data)

correct = 0

for idx, row in enumerate(validation.data):
    author = validation.authors[idx]
    X, y = train.__generateAuthorData__(author)
    classifier.fit(X[:, features], y)

    if classifier.predict(row) == author:
        correct += 1

results = correct / len(validation.data)

print('Result, with K={} and p={}:\
        {}'.format(args.K, args.p, results))
