# -*- coding: utf-8 -*-
import os
import sys
import fileinput


def Convert(x):
    c = {
        'aren’t': 'are not',
        "can't": 'cannot',
        "couldn't": 'could not',
        "didn't": 'did not',
        "hasn't": 'has not',
        "haven't": 'have not',
        "isn't": 'is not',
        "mustn't": 'must not',
        "shan't": 'shall not',
        "shouldn't": 'should not',
        "wasn't": 'was not',
        "weren't": 'were not',
        "won't": 'will not',
        "wouldn't": 'would not',
        'data-set': 'dataset',
        'hyper-parameters': 'hyperparameters',
        'n-grams': 'N-grams',
        'n-gram': 'N-gram'
    }

    for key in c:
        x = x.replace(key, c[key])

        cap_key = key[0].upper() + key[1:]
        cap_val = c[key][0].upper() + c[key][1:]

        x = x.replace(cap_key, cap_val)

    return x


reportFolder = sys.argv[-1]

d = os.listdir(reportFolder)

for i in filter(lambda x: x.split('.')[-1] == 'tex', d):
    with fileinput.FileInput(
            reportFolder + i, inplace=True, backup='.bak') as f:
        for line in f:
            print(Convert(line), end='')
