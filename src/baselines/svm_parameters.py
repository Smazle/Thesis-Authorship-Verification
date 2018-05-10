#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from collections import Counter
import pandas as pd
import pickle
import time

# Set random state for reproducible results.
np.random.seed(1337)
random.seed(1337)

parser = argparse.ArgumentParser(
    description='Use features in feature file specified by the boolean ' +
                'vector given to find best svm parameters and test the svm ' +
                'on part of the data.'
)
parser.add_argument(
    'featurefile',
    type=str,
    help='Path to file containing features'
)
parser.add_argument(
    'features',
    type=str,
    help='File contaning the product of the feature selection',
    default=None,
    nargs='?'
)

args = parser.parse_args()

# Load data.
print('Loading Data')
with open(args.featurefile, 'r', encoding='utf8') as feature_file:
    data = pd.read_csv(feature_file, )
    authors = data.as_matrix(columns=['author']).flatten()

    datacolumns = filter(lambda x: x != 'author', data.columns)
    X = data.as_matrix(columns=datacolumns)
    total_text_number = X.shape[0]
    total_feature_number = X.shape[1]
print('Done')

# Get list of unique authors in a scrambled order so we don't have any bias in
# the order of the datafile.
unique_authors = np.unique(authors)
np.random.shuffle(unique_authors)

training_authors = unique_authors

# Read which features we should train on.
if args.features is not None:
    with open(args.features, 'r') as f:
        data = pd.read_csv(f, header=None)
        indices = data.as_matrix(columns=[data.columns[0]]).flatten()
        accuracies = data.as_matrix(columns=[data.columns[1]]).flatten()

        feature_n = np.argmax(accuracies)
        feature_set = indices[0:feature_n]

        features_to_use = np.zeros((total_feature_number, ), dtype=np.bool)
        features_to_use[feature_set] = 1

    X = X[:, features_to_use]

# Find the best hyperparameters using the training authors.
C_range = [float(10**x) for x in range(-16, 0, 2)]
gamma_range = [float(10**x) for x in range(-3, 9, 2)]
best_params = Counter()
res = {}
pickle.dump(res, open('Pick.p', 'wb'))
length = len(training_authors)
for idx, author in enumerate(training_authors):
    t = time.time()
    # Split texts into those written by same and different authors.
    same_author = X[authors == author]
    different_author = X[authors != author]
    same_author_n = same_author.shape[0]

    print('handling author', author, 'with', same_author_n, 'texts')

    # Draw opposition of same size as the number of texts the author has.
    indices = np.random.choice(different_author.shape[0], same_author_n,
                               replace=False)
    random_author = different_author[indices, :]

    # Define dataset we are training on.
    X_train = np.vstack([same_author, random_author])
    y_train = np.array([1] * same_author_n + [0] * same_author_n)

    # Leave one out cross validation over C and gamma range.
    cv = StratifiedKFold(3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    try:
        grid.fit(X_train, y_train)
    except ValueError:
        print('Skipped, length, ', same_author_n)
        continue

    print('\t', 'best params', grid.best_params_)
    print('\t', 'best score', grid.best_score_)
    print('\t', 'Time', time.time() - t)
    print('\t', 'Author Index', '{}/{}'.format(idx, length))

    best_C_gamma = (grid.best_params_['C'], grid.best_params_['gamma'])
    best_params = best_params + Counter([best_C_gamma])

    scores = grid.cv_results_['mean_test_score']
    params = grid.cv_results_['params']
    for s, p in zip(scores, params):
        c_gamma = (p['C'], p['gamma'])

        if c_gamma not in res:
            res[c_gamma] = [s]
        else:
            res[c_gamma].append(s)

((C, gamma), count) = best_params.most_common()[0]
print(best_params.most_common())

print('final best parameters', 'C', C, 'gamma', gamma)
print('Score', np.mean(res[(C, gamma)]))
pickle.dump(res, open('Pick.p', 'wb'))

output = [str((key, np.mean(value))) for key, value in res.items()]
print('\n'.join(output))
