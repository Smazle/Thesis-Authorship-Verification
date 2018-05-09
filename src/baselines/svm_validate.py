#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from .feature_search import FeatureSearch
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold


parser = argparse.ArgumentParser(
    description='Uses the selected features and parameters, \
                to compute the validation error of the training'
)

parser.add_argument(
    'validationFile',
    type=str,
    help='Path to file containing raw features'
)

parser.add_argument(
    'features',
    type=str,
    help='File containing the product of the feature selection'
)

parser.add_argument(
    '--C',
    type=float,
    help='The parameter selected value of C'
)

parser.add_argument(
    '--gamma',
    type=float,
    help='The parameter selected value of gamma'
)

args = parser.parse_args()

classifier = SVC(C=args.C, gamma=args.gamma, kernel='rbf')

with open(args.features, 'r') as f:
    data = pd.read_csv(f, header=None)
    indices = data.as_matrix(columns=[data.columns[0]]).flatten()
    accuracies = data.as_matrix(columns=[data.columns[1]]).flatten()

    feature_n = np.argmax(accuracies)
    feature_set = indices[0:feature_n]

fs = FeatureSearch(None, None, authorLimit=None, normalize=False, validator=StratifiedKFold(3))
fs.__generateData__(args.validationFile)

output = fs.__evaluate_classifier__(classifier, feature_set)

print('Validation Result, with C={} and gamma={}:\
        {}'.format(args.C, args.gamma, output))
