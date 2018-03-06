# -*- coding: utf-8 -*-
import numpy as np
import sys
# import os
import subprocess
import pickle

inp = sys.argv[1]

base = './extended_delta.py ../feature_extraction/output \
        --opposing-file ../feature_extraction/output '

featureCount = np.loadtxt(inp, dtype=str).shape[1]

neigbors = range(2, 11)


neigbor_results = {}

prev = (0, 0)
for count in neigbors:
    features = [str(i) for i in range(0, featureCount-1)]
    runBase = base + '--opposing-set-size ' + str(count) + ' '

    changed = False

    while not changed:

        for i in features:
            features.pop(int(i))

            command = runBase + '--features ' + ' '.join(features)
            result = float(str(subprocess.check_output(
                command, shell=True)).replace('\n', ''))
            # import pdb; pdb.set_trace()

            features.insert(int(i), i)

            if prev[1] < result:
                prev = (int(i), result)
                changed = True
                print(prev)

        features.pop(prev[0])
        changed = False

    neigbor_results[neigbors] = (prev[1], features)

pickle.dump(neigbor_results, open('params', 'wb'))
