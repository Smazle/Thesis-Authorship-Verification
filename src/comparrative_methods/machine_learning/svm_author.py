#!/usr/bin/python3

import argparse
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut


# Set seed to make sure we get reproducible results.
np.random.seed(7)


def single_one(n, size, dtype=np.bool):
    x = np.zeros((size, ), dtype=dtype)
    x[n] = 1
    return x


def analyse_header(header):
    change_indices = []
    prev = None
    columns = 0
    for i, value in enumerate(header.rstrip().split(';')):
        columns = columns + 1
        if prev != value:
            change_indices.append(i)
            prev = value

    feature_classes = {}
    for (start, end) in zip(change_indices, change_indices[1:]):
        feature_classes[start] = list(range(start, end))

    return feature_classes


def get_next_missing(feature_classes):
    missing = []

    for key in feature_classes:
        values = feature_classes[key]
        if len(values) != 0:
            missing.append(values[0])

    return missing


def remove_feature(feature_classes, value):
    for key in feature_classes:
        values = feature_classes[key]

        if value in values:
            values.remove(value)


parser = argparse.ArgumentParser(
    description='Find best features from feature file via a greedy search')
parser.add_argument(
    'features',
    type=str,
    help='Path to file containing features'
)
args = parser.parse_args()

with open(args.features, 'r') as feature_file:
    header = feature_file.readline()  # Skip first line.
    feature_classes = analyse_header(header)
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

unique_authors = np.sort(np.unique(authors))

# While we keep improving accuracy continue.
current_features = np.zeros((X.shape[1], ), dtype=np.bool)
prev_best = 0.0
while True:
    # Try to add each of the missing features and keep the version that
    # improves score the most.
    best_index = None
    for missing in get_next_missing(feature_classes):
        print('Testing', missing)
        new_feature = single_one(missing, current_features.shape[0])
        features = X[:, np.logical_or(current_features, new_feature)]

        scores = []
        for author in unique_authors:
            author_texts = features[authors == author]
            other_texts = features[authors != author]

            opposition = other_texts[np.random.choice(
                other_texts.shape[0],
                author_texts.shape[0],
                replace=False)]

            # TODO: Change C and gamma values.
            classifier = SVC(kernel='rbf', C=100, gamma=0.00001)
            X_train = np.vstack([author_texts, opposition])
            y_train = np.array([1] * author_texts.shape[0] + [0] * author_texts.shape[0])

            cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=0)
            score = cross_val_score(classifier, X_train, y_train, cv=cv)

            scores.append(np.mean(score))

        if np.mean(scores) > prev_best:
            prev_best = np.mean(scores)
            best_index = missing

    if best_index is None:
        # We are done.
        break
    else:
        print('prev_best', prev_best, 'best_index', best_index)
        current_features[best_index] = True
        remove_feature(current_features, best_index)

    np.savetxt('best_features.npz', current_features)
