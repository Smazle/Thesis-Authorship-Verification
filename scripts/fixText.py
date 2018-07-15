# -*- coding: utf-8 -*-
import os
import sys
import fileinput
from nltk.tokenize import word_tokenize


def Convert(line, x):
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
        'N-grams': 'n-grams',
        'N-gram': 'n-gram',
        'TPS': 'TP',
        'TNS': 'TN',
        'FNS': 'FN',
        'FPS': 'FP',
        'Gradient Descent': 'gradient descent',
        'base-line': 'baseline',
        'base-lines': 'baselines',
    }

    for key in c:

        x = x.replace(key, c[key])

        cap_key = key[0].upper() + key[1:]
        cap_val = c[key][0].upper() + c[key][1:]

        x = x.replace(cap_key, cap_val)

    return x


def Warnings(line, x):
    warnings = ["'s"]

    for key in warnings:
        if key in x:
            print(key, line, x)

    word = word_tokenize(x)
    for i in range(len(word) - 1):
        if len(word[i][0]) < 2:
            continue

        if word[i][0].isalnum():
            if word[i][0] == word[i][0].upper() and \
                    word[i + 1][0] == word[i + 1][0].upper():
                print(word[i][0], line, x)


reportFolder = sys.argv[-1]

d = os.listdir(reportFolder)

for i in filter(lambda x: x.split('.')[-1] == 'tex', d):
    with fileinput.FileInput(
            reportFolder + i, inplace=True, backup='.bak') as f:
        for idx, line in enumerate(f):
            print(Convert(idx, line), end='')

for i in filter(lambda x: x.split('.')[-1] == 'tex', d):
    with open(reportFolder + i, 'r') as f:
        print(i)
        for idx, line in enumerate(f):
            Warnings(idx + 1, line)
        input()
