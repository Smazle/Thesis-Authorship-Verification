#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .feature_search import FeatureSearch
from sklearn.model_selection import StratifiedKFold
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
    type=float,
    help='Number of unique authors to use',
    default=.5
)

args = parser.parse_args()

svm = SVC()
# TODO: This should not be the LeaveOneOut cross validation since that does not
# work for SVM's.
search = FeatureSearch(svm, args.features, args.authors, normalize=False,
                       validator=StratifiedKFold(3))
search.fit(args.datafile, args.outfile)
