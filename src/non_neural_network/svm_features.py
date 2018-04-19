#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .feature_search import FeatureSearch
from sklearn.svm import SVC
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
    help='Path to file to save features in.'
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
    type=int,
    help='Number of unique authors to use',
    default=100
)

args = parser.parse_args()

svm = SVC()
search = FeatureSearch([svm], args.features, args.authors)
search.fit(args.datafile, args.outfile)
