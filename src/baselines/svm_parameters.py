#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from collections import Counter
import pandas as pd

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
    help='Path to file containing boolean vector to specify which features ' +
         'to use.'
)
parser.add_argument(
    'training_validation_split',
    type=float,
    help='Number between 0 and 1 specifying how much data should be ' +
         'reserved for validation and not for finding hyperparameters.'
)
args = parser.parse_args()

# Load data.
with open(args.featurefile, 'r') as feature_file:
    data = pd.read_csv(feature_file)
    authors = data.as_matrix(columns=['author']).flatten()

    datacolumns = filter(lambda x: x != 'author', data.columns)
    X = data.as_matrix(columns=datacolumns)
    total_text_number = X.shape[0]
    total_feature_number = X.shape[1]

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
    data = pd.read_csv(f, header=None)
    indices = data.as_matrix(columns=[data.columns[0]]).flatten()
    accuracies = data.as_matrix(columns=[data.columns[1]]).flatten()

    feature_n = np.argmax(accuracies)
    feature_set = indices[0:feature_n]

    features_to_use = np.zeros((total_feature_number, ), dtype=np.bool)
    features_to_use[feature_set] = 1

X = X[:, features_to_use]

# Find the best hyperparameters using the training authors.
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
    gamma_range = np.logspace(-7, 5, 7)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = LeaveOneOut()
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    print('\t', 'best params', grid.best_params_)
    print('\t', 'best score', grid.best_score_)

    best_C_gamma = (grid.best_params_['C'], grid.best_params_['gamma'])
    best_params = best_params + Counter([best_C_gamma])

((C, gamma), count) = best_params.most_common()[0]

print('final best parameters', 'C', C, 'gamma', gamma)

# Train an svm for each validation author using the best parameters found.
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

        # Single prediction since we use leave one out.
        prediction = model.predict(X_train[test_index])

        if prediction[0] == y_train[test_index][0]:
            author_scores.append(1.0)
        else:
            author_scores.append(0.0)

    scores.append(np.mean(author_scores))

print('final score', np.mean(scores))
