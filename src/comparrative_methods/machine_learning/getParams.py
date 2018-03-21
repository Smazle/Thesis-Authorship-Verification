# -*- coding: utf-8 -*-
import numpy as np
import sys
import pickle
import extended_delta as ed
import argparse

inp = sys.argv[1]


def increment(features, startIndex):
    for i in range(len(features)):
        m = max(features[i]) + 1
        if m not in startIndex.keys():
            features[i].append(m)

    return features


base = './extended_delta.py ../feature_extraction/output \
        --opposing-file ../feature_extraction/output '


data = np.loadtxt(inp, dtype=np.float, delimiter=' ', skiprows=1)

featureNames = [tuple(x.split(' '))
                for x in open(inp).readline().rstrip().split(';')]
featureCount = len(featureNames)

# print(featureNames)

startIndex = {key: featureNames.index(key) for key in list(set(featureNames))}


print(startIndex)


neigbors = range(2, 11)

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=int)
parser.add_argument('--opposing-set-size', type=int)

neigbor_results = {}

prev = (0, 0)
for count in neigbors:
    print('Neighbors', count)
    runBase = base + '--opposing-set-size ' + str(count) + ' '

    changed = False

    indexes = sorted(startIndex.values())
    maxDiff = max([abs(indexes[i - 1] - x) for i, x in enumerate(indexes)][1:])
    features = [[x] for x in indexes]

    fallCount = 0
    for i in range(maxDiff):

        args = parser.parse_args(
            ['--metric', '1', '--opposing-set-size', str(count)])
        result = ed.runMe(args, data, data, np.concatenate(features))
        # import pdb; pdb.set_trace()

        if prev[1] < result:
            prev = (np.concatenate(features), result)
            fallCount = 0
            print(result, 'Changed')
        else:
            fallCount += 1
            print(result)

        if fallCount == 3:
            print(prev)
            open('ExtendedParams.features', 'a').write(str(prev) + '\n')
            break

        features = increment(features, startIndex)
        print(features)

    neigbor_results[neigbors] = (prev[1], np.concatenate(features))

pickle.dump(neigbor_results, open('params', 'wb'))
