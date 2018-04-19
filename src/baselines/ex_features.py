#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .feature_search import FeatureSearch
from sklearn.neighbors import KNeighborsClassifier
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='Run different kind of features selection'
)

parser.add_argument(
    'datafile',
    type=str,
    help='Path to feature file',
)

parser.add_argument(
    'outfile',
    type=str,
    help='Path to file to write features in.'
)

parser.add_argument(
    '--features',
    type=int,
    help='The minimum number of features to fetch, \
            only relevant if single=True',
    default=50
)

parser.add_argument(
    '--authors',
    type=float,
    help='Number of unique authors to use',
    default=.5
)

args = parser.parse_args()

knn = KNeighborsClassifier(3)
search = FeatureSearch(knn, args.features, args.authors, normalize=True)
search.fit(args.datafile, args.outfile)
