#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import random


# Set seed to make sure we get reproducible results.
np.random.seed(7)
random.seed(7)


parser = argparse.ArgumentParser(
    description='Find best features from feature file via a greedy search')
parser.add_argument(
    'features',
    type=str,
    help='Path to file containing features'
)
parser.add_argument(
    '--authors',
    type=int,
    help='How many authors to use for the search. If none then all are used.'
)
args = parser.parse_args()

# Load data.
with open(args.features, 'r') as feature_file:
    data = pd.read_csv(feature_file)
    authors = data.as_matrix(columns=['author']).flatten()

    datacolumns = filter(lambda x: x != 'author', data.columns)
    X = data.as_matrix(columns=datacolumns)

unique_authors = np.sort(np.unique(authors))

# If we are given an amount of authors only use them.
if args.authors is not None:
    unique_authors = unique_authors[0:args.authors]
    X = X[np.isin(authors, unique_authors)]
    authors = authors[np.isin(authors, unique_authors)]

# While we improve accuracy continue.
best_features = np.zeros((X.shape[1], ), dtype=np.bool)
prev_best = 0.0
while True:
    best_index = None
    for i in np.nonzero(np.logical_not(best_features))[0]:
        print('Trying feature', i)
        assert not best_features[i]
        best_features[i] = True
        features = X[:, best_features]
        best_features[i] = False

        scores = []
        for author in unique_authors:
            author_texts = features[authors == author]
            other_texts = features[authors != author]

            opposition = other_texts[np.random.choice(
                other_texts.shape[0],
                author_texts.shape[0],
                replace=False)]

            classifier = SVC(kernel='rbf', C=100, gamma=1000.0)
            X_train = np.vstack([author_texts, opposition])
            y_train = np.array(
                [1] * author_texts.shape[0] +
                [0] * author_texts.shape[0]
            )

            score = cross_val_score(classifier, X_train, y_train)

            scores.append(np.mean(score))

        if np.mean(scores) > prev_best:
            print('\tFound better with score', np.mean(score))
            prev_best = np.mean(scores)
            best_index = i

    np.savetxt('best_features.npz', best_features)

    if best_index is None:
        print('Best index is none so this is the best we can do')
        break
    else:
        best_features[best_index] = True
