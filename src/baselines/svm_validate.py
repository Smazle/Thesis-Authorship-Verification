#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.svm import SVC

parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training')

parser.add_argument(
    'validationFile', type=str, help='Path to file containing raw features')

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection')

parser.add_argument(
    '--C', type=float, help='The parameter selected value of C')

parser.add_argument(
    '--gamma', type=float, help='The parameter selected value of gamma')

args = parser.parse_args()

classifier = SVC(C=args.C, gamma=args.gamma, kernel='rbf')

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

feature_obj = FeatureSearch(None, None, authorLimit=None, normalize=False)
feature_obj.__generateData__(args.validationFile)

correct = 0

for idx, row in enumerate(feature_obj.data):
    author = feature_obj.authors[idx]

    X, y = feature_obj.getAuthorData(author, row)
    classifier.fit(X[:, features], y)

    if classifier.predict(row[features].reshape(1, -1)) == author:
        correct += 1

results = correct / len(feature_obj.data)

print('Validation Result, with C={} and gamma={}:\
        {}'.format(args.C, args.gamma, results))
