#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.svm import SVC
import pandas as pd

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

parser.add_argument(
    '--C', type=float, help='The parameter selected value of C')

parser.add_argument(
    '--gamma', type=float, help='The parameter selected value of gamma')

args = parser.parse_args()

classifier = SVC(C=args.C, gamma=args.gamma, kernel='rbf')

with open(args.features, 'r') as f:
    data = pd.read_csv(f, header=None)
    indices = data.as_matrix(columns=[data.columns[0]]).flatten()
    accuracies = data.as_matrix(columns=[data.columns[1]]).flatten()

    feature_n = np.argmax(accuracies)
    feature_set = indices[0:feature_n]

train = FeatureSearch(None, None, authorLimit=None, normalize=False)
train.__generateData__(args.trainingFile)

validation = FeatureSearch(None, None, authorLimit=None, normalize=False)
validation.__generateData__(args.validationFile)

correct = 0

for idx, row in enumerate(validation.data):
    author = validation.author[idx]
    X, y = train.__generateData__(author)
    classifier.fit(X[:, feature_set], y)

    if classifier.predict(row) == author:
        correct += 1

results = correct / len(validation.data)

print('Validation Result, with C={} and gamma={}:\
        {}'.format(args.C, args.gamma, results))
