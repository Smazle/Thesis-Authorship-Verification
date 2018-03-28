#!/usr/bin/env python3

import argparse
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Set seed to make sure we get reproducible results.
np.random.seed(7)


parser = argparse.ArgumentParser(
    description='Find best features from feature file via a greedy search')
parser.add_argument(
    'features',
    type=str,
    help='Path to file containing features'
)
args = parser.parse_args()

with open(args.features, 'r') as feature_file:
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

unique_authors = np.sort(np.unique(authors))

scores = []
for author in unique_authors:
    print('Testing author', author)
    author_texts = X[authors == author]
    other_texts = X[authors != author]

    opposition = other_texts[np.random.choice(
        other_texts.shape[0],
        author_texts.shape[0],
        replace=False)]

    # TODO: Change C and gamma values.
    classifier = SVC(kernel='rbf', C=100, gamma=0.00001)
    X_train = np.vstack([author_texts, opposition])
    y_train = np.array([1] * author_texts.shape[0] + [0] * author_texts.shape[0])

    score = cross_val_score(classifier, X_train, y_train)

    scores.append(np.mean(score))

print(scores)
print(np.mean(scores))
