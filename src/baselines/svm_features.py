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

args = parser.parse_args()

svm = SVC()
search = FeatureSearch(svm, args.features, authorLimit=None, normalize=False,
                       validator=StratifiedKFold(3))
search.fit(args.datafile, args.outfile)
