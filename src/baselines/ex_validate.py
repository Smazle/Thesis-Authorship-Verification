#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.neighbors import KNeighborsClassifier
import pickle
import random
import sys

parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training')

parser.add_argument(
    'validation_file', type=str, help='Path to file containing raw features')

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection')

parser.add_argument(
    '--scaler', type=str, help='Path to pickled scaler object', required=True)

parser.add_argument(
    '--K', type=int, help='The parameter selected value of K', required=True)

parser.add_argument(
    '--p', type=int, help='The parameter selected value of p', required=True)

parser.add_argument(
    '--negative-chance',
    type=float,
    help='With what chance to generate a negative sample.',
    required=True)

args = parser.parse_args()

# Load the indices of the features we should use.
features = np.loadtxt(args.features, dtype=np.str, delimiter=',')
accuracies = features[:, 1].astype(np.float)
features = features[:, 0].astype(np.int)
features = features[0:np.argmax(accuracies)]

# Loop through unique authors and evaluate performance of his/her newest text.
feature_search = FeatureSearch(None, None, authorLimit=None, normalize=False)
feature_search.__generateData__(args.validation_file)

# Scale data according to training.
with open(args.scaler, 'rb') as f:
    scaler = pickle.load(f)
    feature_search.data = scaler.transform(feature_search.data)

classifier = KNeighborsClassifier(n_neighbors=args.K, p=args.p)

tps = 0
tns = 0
fps = 0
fns = 0
positives = 0
negatives = 0
for author in np.unique(feature_search.authors):
    X, y = feature_search.__generateAuthorData__(author)
    X = X[:, features]

    if y.shape == (2, ):
        print('WARNING to few texts', file=sys.stderr)
        continue

    # Predict a positive.
    positives += 1
    X_positives = X[y == 1]
    X_negatives = X[y == 0]
    y_positives = y[y == 1]
    y_negatives = y[y == 0]

    newest_X = X_positives[-1]
    newest_y = y_positives[-1]
    train_X = np.vstack([X_positives[0:-1], X_negatives[0:-1]])
    train_y = np.hstack([y_positives[0:-1], y_negatives[0:-1]])

    classifier.fit(train_X, train_y)
    prediction = classifier.predict(newest_X.reshape(1, -1))[0]

    if prediction == 1:
        tps += 1
    else:
        fns += 1

    # Predict a negative if we are asked to.
    if random.random() <= args.negative_chance:
        negatives += 1
        newest_X = X_negatives[-1]
        newest_y = y_negatives[-1]
        train_X = np.vstack([X_positives[0:-1], X_negatives[0:-1]])
        train_y = np.hstack([y_positives[0:-1], y_negatives[0:-1]])

        classifier.fit(train_X, train_y)
        prediction = classifier.predict(newest_X.reshape(1, -1))[0]

        if prediction == 1:
            fps += 1
        else:
            tns += 1

accuracy = (tps + tns) / (tps + tns + fps + fns)
accusation_error = fns / (fns + tns)

print('positives \t negatives \t tps \t tns \t fps fns' +
      '\t accuracy \t accusation_error')

print(positives, '\t', negatives, '\t', tps, '\t', tns, '\t', fps, '\t', fns,
      '\t', accuracy, '\t', accusation_error)
