#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
import argparse
import numpy as np
import pandas as pd
from .feature_search import FeatureSearch

parser = argparse.ArgumentParser(
    description='Run to determine the best hyperparmeters for\
                 knn classifier'
)

parser.add_argument(
    'featurefile',
    type=str,
    help='Path to file containing raw features'
)

parser.add_argument(
    'features',
    type=str,
    help='Path to the file containing the selected features'
)

parser.add_argument(
    '--K',
    type=int,
    nargs='+',
    help='Which K in KNN to to determine accuracy of, from and\
        to',
    default=[1, 15]
)

args = parser.parse_args()

features = np.loadtxt(args.features, dtype=float, delimiter=',')
features = features[:np.argmax(features, axis=0)[0]][:, 0].astype(int)

fs = FeatureSearch(None, None, authorLimit=False, normalize=True)
fs.__generateData__(args.featurefile)

fs.data = fs.data[:, features]

results = {}
for p in range(1, 6):
    results[p] = {}

    for K in range(args.K[0], args.K[-1] + 2, 2):
        print('Determining Accuracy of K = {}, and p = {}'.format(K, p))

        knn = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=p)
        scores = []
        for author in np.unique(fs.authors):
            X, y = fs.__generateAuthorData__(author)

            try:
                score = cross_val_score(knn, X, y, cv=LeaveOneOut())
                scores.append(np.mean(score))
            except ValueError:
                continue

        if(len(scores) == 0):
            print('No texts can handle K size of {}'.format(K))
            break
        else:
            results[p][K] = {'authors': len(scores), 'score': np.mean(scores)}
            print(results[p][K])
