#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import random
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

np.random.seed(7)
random.seed(7)


def main(args, training=None, data=None, features=None):
    # Import data ([features...], truth, author).

    if data is None:
        data = np.loadtxt(args.file, dtype=str, delimiter=' ', skiprows=1)

    if training is None:
        training = np.loadtxt(args.opposing_file,
                              dtype=str, delimiter=' ', skiprows=1)

    authors = data[:, 0]
    data = data[:, 1:].astype(np.float)

    training_authors = training[:, 0]
    training = training[:, 1:].astype(np.float)

    scaler = StandardScaler().fit(training)
    training = scaler.transform(training)
    data = scaler.transform(data)

    if features is None:
        features = [int(''.join(a)) for a in args.features]
        training = training[:, features].reshape(
            training.shape[0], len(features))
        data = data[:, features].reshape(data.shape[0], len(features))
    else:
        training = training[:, features].reshape(
            training.shape[0], len(features))
        data = data[:, features].reshape(data.shape[0], len(features))

    predictions = []
    for i, author in enumerate(authors):

        enumeration = list(enumerate(training_authors))

        # Find texts written by that author
        own = list(filter(lambda x: x[1] == author and
                          not np.array_equal(training[x[0]], data[i]),
                          enumeration))
        own = [x[0] for x in own]

        # Find texts not written by that author
        opposing = list(filter(lambda x: x[1] != author, enumeration))
        opposing_idx = random.sample(opposing, len(own))
        opposing_idx = [x[0] for x in opposing_idx]

        X = np.append(training[own], training[opposing_idx], axis=0)
        y = np.append(training_authors[own], [0] * len(own), axis=0)

        model = neighbors.KNeighborsClassifier(
            n_neighbors=args.N, weights='uniform', algorithm='auto',
            metric='minkowski', p=args.metric)
        model.fit(X, y)

        try:
            prediction = int(model.predict([data[i]]))
            predictions.append(1 if prediction == author else 0)

            otherAuthor = [training[random.sample(opposing, 1)[0][0]]]
            prediction = int(model.predict(otherAuthor))
            predictions.append(1 if prediction == 0 else 0)
        except ValueError:
            print('Dublcate Text Found, Skipping', len(predictions) / 2)

    result = np.mean(predictions)
    return result


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Run delta method with smaller\
                number of different authors')
    parser.add_argument(
        'file',
        type=str,
        help='Data file location')
    parser.add_argument(
        '--opposing-file',
        type=str,
        help='Training Data File Location',
        default='')
    parser.add_argument(
        '--with-normalization',
        help='Whether or not to normalize data.',
        action='store_true',
        default=False)
    parser.add_argument(
        '--N',
        help='Number of opposing authors to use',
        type=int,
        default=1)
    parser.add_argument(
        '--metric',
        help='Which Minkowski metric to use given 1 it will be the Manhattan' +
        ' distance and 2 it is the Euclidean distance',
        type=int,
        default=1)
    parser.add_argument(
        '--with-PN',
        help='Whether or not to also print True Positive, False Positive,' +
        'True Negative and False Negative.',
        action='store_true',
        default=False)
    parser.add_argument(
        '--features',
        type=list,
        default=[],
        nargs='+'
    )

    args = parser.parse_args()
    print(main(args))
