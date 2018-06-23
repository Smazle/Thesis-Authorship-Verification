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

parser.add_argument(
    '--scaler',
    type=str,
    help='Path to standard scaler fitted to the training data')

parser.add_argument('--model', type=str, help='Path to pickled model')

args = parser.parse_args()

classifier = KNeighborsClassifier(n_neighbors=args.K, p=args.p)

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

if args.model is None or args.scaler is None:
    train = FeatureSearch(None, None, None)
    train.__generateData__(args.trainingFile)

    X = y = []
    for author in np.unique(train.authors):
        new_x, new_y = train.__generateAuthorData(author)
        X += new_x
        y += new_y

    classifier.fit(X[:, features], y)
    pickle.dump(classifier, open('Model_ED.p', 'wb'))
    scaler = train.scaler

if args.model is not None and args.scaler is not None:
    scaler = pickle.loads(open(args.scaler, 'rb'))
    model = pickle.loads(open(args.model, 'rb'))

validation = FeatureSearch(None, None, None)
validation.__generateData__(args.validationFile)
validation.data = scaler.transform(validation.data)

X = y = []
for author in np.unique(validation.authors):
    new_x, new_y = validation.__generateAuthorData(author)
    X += new_x
    y += new_y

results = classifier.predict(X)
results = sum(np.equal(results, y)) / len(results)

print('Result, with K={} and p={}:\
        {}'.format(args.K, args.p, results))
