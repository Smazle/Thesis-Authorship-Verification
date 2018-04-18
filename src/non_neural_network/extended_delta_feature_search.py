#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import extended_delta as ed
import numpy as np
import sys


parser = argparse.ArgumentParser(
    description='Run different kind of features selection'
)

parser.add_argument(
    'file',
    type=str,
    help='Path to feature file',
    default=''
)

parser.add_argument(
    '--single',
    type=bool,
    help='Decides wether to check all features one a time',
    default=True,
)

parser.add_argument(
    '--count',
    type=int,
    help='The minimum number of features to fetch, \
            only relevant if single=True',
    default=50
)

args = parser.parse_args()

featureFile = args.file


def increment(features, startIndex):
    for i in range(len(features)):
        m = max(features[i]) + 1
        if m not in startIndex.keys():
            features[i].append(m)

    return features


base = './extended_delta.py' + featureFile + \
    ' --opposing-file' + featureFile + ' '


data = np.loadtxt(featureFile, dtype=str, delimiter=',',
                  skiprows=1, encoding='utf-8')


featureNames = [tuple(x.split('-'))
                for x in open(featureFile).readline().rstrip().split(',')]
featureCount = len(featureNames)

startIndex = {key: featureNames.index(key)
              for key in list(set(featureNames[1:]))}


print(startIndex)


neighbors = range(1, 10)

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=int)
parser.add_argument('--N', type=int)


if not args.single:

    for count in neighbors:
        prev = (0, 0)

        changed = False

        indexes = sorted(startIndex.values())
        maxDiff = max([abs(indexes[i - 1] - x)
                       for i, x in enumerate(indexes)][1:])
        features = [[x] for x in indexes]

        args = parser.parse_args(['--metric', '1', '--N', str(count)])
        print('Neighbors', args.N)

        fallCount = 0
        for i in range(maxDiff):
            data_copy = np.copy(data)
            result = ed.main(args, data_copy, data_copy,
                             np.concatenate(features))

            if prev[1] < result:
                prev = (np.concatenate(features), result)
                fallCount = 0
                print(result, 'Changed')
            else:
                fallCount += 1
                print(result)

            if fallCount == 3:
                print(prev)
                open('ExtendedParams.features', 'a')\
                    .write(','.join(list(map(str, prev[0])) +
                                    [str(prev[1])]) + '\n')
                break

            print(features)
            features = increment(features, startIndex)
else:

    for N in neighbors:
        prev = (0, 0)
        features = []

        args = parser.parse_args(['--metric', '1', '--N', str(count)])

        for i in range(args.count):
            data_copy = np.copy(data)
            results = [(idx, ed.main(args, data_copy, data_copy,
                                     features + [idx])) for idx
                       in range(featureCount)
                       if idx not in features]

            maxRes = max(result, key=lambda x: x[1])
            features += [maxRes[0]]
            print(N, maxRes, features, '\n')

        open('ExtendedParams.L1.Single', 'a').write(
            ', '.join([N] + features) + '\n')
