#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.svm import SVC
import pandas as pd
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

parser.add_argument(
    '--C', type=float, help='The parameter selected value of C')

parser.add_argument(
    '--gamma', type=float, help='The parameter selected value of gamma')

parser.add_arumgent('--model', type=str, help='Path to pickled model')

args = parser.parse_args()

classifier = SVC(C=args.C, gamma=args.gamma, kernel='rbf')

with open(args.features, 'r') as f:
    data = pd.read_csv(f, header=None)
    indices = data.as_matrix(columns=[data.columns[0]]).flatten()
    accuracies = data.as_matrix(columns=[data.columns[1]]).flatten()

    feature_n = np.argmax(accuracies)
    feature_set = indices[0:feature_n]

if args.model is None:
    train = FeatureSearch(None, None, authorLimit=None, normalize=False)
    train.__generateData__(args.trainingFile)

    X = y = []
    for author in np.unique(train.authors):
        new_x, new_y = train.__generateAuthorData(author)
        X += new_x
        y += new_y

    classifier.fit(X[:, feature_set], y)
    pickle.dump(classifier, open('Model_SVM.p', 'wb'))
else:
    pickle.load(open(args.model, 'rb'))

validation = FeatureSearch(None, None, authorLimit=None, normalize=False)
validation.__generateData__(args.validation)

X = y = []
for author in np.unique(validation.authors):
    new_x, new_y = validation.__generateAuthorData(author)
    X += new_x
    y += new_y

results = classifier.predict(X[:, feature_n])
results = sum(np.equal(results, y)) / len(results)

print('Validation Result, with C={} and gamma={}:\
        {}'.format(args.C, args.gamma, results))
