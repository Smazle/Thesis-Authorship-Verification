#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import random
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


np.random.seed(7)


def runMe(args, training=None, data=None, features=None):
    # Import data ([features...], truth, author).

    if data is None:
        data = np.loadtxt(args.file, dtype=np.float, delimiter=' ', skiprows=1)

    if training is None:
        training = np.loadtxt(args.opposing_file,
                              dtype=np.float, delimiter=' ', skiprows=1)

    authors = data[:, -1].astype(np.int)
    data = np.array(data[:, :-1])
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)

    training_authors = training[:, -1].astype(np.int)
    training = np.array(training[:, :-1])

    if features is None:
        features = [int(''.join(a)) for a in args.features]
        training = training[:, features].reshape(
            training.shape[0], len(features))
        data = data[:, features].reshape(data.shape[0], len(features))
    else:
        training = training[:, features].reshape(
            training.shape[0], len(features))
        data = data[:, features].reshape(data.shape[0], len(features))

    #   Temporary, to be removed
    #   temp = list(range(0, int(training.shape[0]/2)))
    #   training[:, -1] = temp*2
    #   authors = training[:, -1].astype(np.int)

    predictions = []
    authorIndex = {key: 0 for key in list(set(training_authors))}
    for i in range(data.shape[0]):
        opposing_index = list(filter(lambda x: x[1] != authors[i],
                                     enumerate(training_authors)))
        opposing_index = random.sample(opposing_index, args.opposing_set_size)
        opposing_index = [x[0] for x in opposing_index]

        temp = list(enumerate(training_authors))
        own = list(filter(lambda x: x[1] == authors[i], temp))

        if authors[i] in authorIndex.keys():
            del own[authorIndex[authors[i]]]
            authorIndex[authors[i]] += 1

        inp = np.append(training[opposing_index], [
                        training[own[0][0]]], axis=0)
        res = np.append(training_authors[opposing_index],
                        [authors[i]], axis=0)

        model = neighbors.KNeighborsClassifier(
            n_neighbors=1, weights='uniform', algorithm='auto',
            metric='minkowski', p=args.metric)
        model.fit(inp, res)

        prediction = int(model.predict([data[i]])[0])
        predictions.append(1 if prediction == authors[i] else 0)

    result = np.sum(predictions) / data.shape[0]
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
        '--opposing-set-size',
        help='Number of opposing authors to use',
        type=int,
        default=5)
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
    print(runMe(args))
