#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from collections import Counter


parser = argparse.ArgumentParser(
    description='Use features in feature file specified by the boolean ' +
        'vector given to find best svm parameters and test the svm on part ' +
        'of the data.'
)
parser.add_argument(
    'featurefile',
    type=str,
    help='Path to file containing features'
)
parser.add_argument(
    'features',
    type=str,
    help='Path to file containing boolean vector to specify which features ' +
        'to use.'
)
parser.add_argument(
    'training_validation_split',
    type=float,
    help='Number between 0 and 1 specifying how much data should be reserved ' +
        'for validation and not for finding hyperparameters.'
)
args = parser.parse_args()

# Read datafile without consuming all my RAM (numpy...).
with open(args.featurefile, 'r') as feature_file:
    feature_file.readline()  # Skip first line.
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    # Number of features is number of columns minus the author column.
    feature_n = len(reader.__next__()) - 1

    line_n = 1
    for line in reader:
        line_n = line_n + 1

    X = np.zeros((line_n, feature_n), dtype=np.float)
    authors = np.zeros((line_n, ), dtype=np.int)

    # Go back to start of file and read again.
    feature_file.seek(0)
    feature_file.readline()
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    for i, line in enumerate(reader):
        X[i] = np.array(list(map(lambda x: float(x), line[0:-1])))
        authors[i] = int(line[-1])

# Get list of unique authors in a scrambled order so we don't have any bias in
# the order of the datafile.
unique_authors = np.unique(authors)
np.random.shuffle(unique_authors)
unique_authors_n = unique_authors.shape[0]

# Get training/validation split in authors.
split = int(args.training_validation_split * unique_authors_n)
training_authors = unique_authors[:split]
validation_authors = unique_authors[split:]

# Read which features we should train on.
with open(args.features, 'r') as f:
    features_to_use = np.loadtxt(f).astype(np.bool)

X = X[:, features_to_use]

best_params = Counter()
for author in training_authors:
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
    C_range = np.logspace(-2, 10, 7)
    gamma_range = np.logspace(-9, 3, 7)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = LeaveOneOut()
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    print('\t', 'best params', grid.best_params_)
    print('\t', 'best score', grid.best_score_)

    best_C_gamma = (grid.best_params_['C'], grid.best_params_['gamma'])
    best_params = best_params + Counter([best_C_gamma])

print(best_params.most_common(1))
print(best_params)

((C, gamma), count) = best_params.most_common()[0]

print(C, gamma)

# Train an svm using the best parameters found.
scores = []
for author in validation_authors:
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

    # Cross validate via leave one out cross validation.
    author_scores = []
    cv = LeaveOneOut()
    for train_index, test_index in cv.split(X_train):
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train[train_index], y_train[train_index])

        predictions = model.predict(X_train[test_index])

        # print(predictions)
        print(y_train[test_index])
        print(predictions - y_train[test_index])
        print()
