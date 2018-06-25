#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.neighbors import KNeighborsClassifier
import pickle

parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training')

parser.add_argument(
    'validationFile', type=str, help='Path to file containing raw features')

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection')

parser.add_argument('--scaler', type=str, help='Path to pickled scaler object')

parser.add_argument('--K', type=int, help='The parameter selected value of K')

parser.add_argument('--p', type=int, help='The parameter selected value of p')

args = parser.parse_args()

classifier = KNeighborsClassifier(n_neighbors=args.K, p=args.p)
scaler = pickle.load(open(args.scaler, 'rb'))

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

feature_obj = FeatureSearch(None, None, None)
feature_obj.__generateData__(args.validationFile)
feature_obj.data = scaler.transform(feature_obj.data)

correct = 0

for idx, row in enumerate(feature_obj.data):
    print(idx, len(feature_obj.data))
    author = feature_obj.authors[idx]
    row = np.array(row)

    X, y = feature_obj.getAuthorData(author, row)

    print(y)
    classifier.fit(X[:, features], y)

    print(author, classifier.predict(row[features].reshape(1, -1)))
    if classifier.predict(row[features].reshape(1, -1)) == author:
        correct += 1

print(correct)
results = correct / float(len(feature_obj.data))

print('Result, with K={} and p={}:\
        {}'.format(args.K, args.p, results))
