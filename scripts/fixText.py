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
        "doesn't": 'does not',
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
        'feed-forward': 'feed forward',
        'over-fitting': 'overfitting',
        'over-fit': 'overfit',
        'training-time': 'training time',
        'back-propagation': 'backpropagation',
        'back-propagate': 'backpropagate',
        'feature-set': 'feature set',
        'testset': 'test set',
        'MaComs': "MaCom's",
        'ghost writer': 'ghostwriter',
        'ghost written': 'ghostwritten',
        'ghost writing': 'ghostwriting'
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

    exceptions = [
        'Figure', 'Figures', 'Danish', 'Section', 'Sections', 'Table', 'Rom',
        'Rome', 'Appendix'
    ]

    if not any(a in x for a in ['}', '{', ']', '[']):
        words = word_tokenize(x)
        for w in words[1:]:
            if w[0].isupper():
                idx = x.index(w)

                if x[idx - 2] != '.' and \
                        not x[idx + 1].isupper() and \
                        x[idx + 1].isalpha() and \
                        w not in exceptions and \
                        x[idx - 2] != '?' and \
                        x[idx - 2] != '!':

                    print(w, line, x)


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
            words = word_tokenize(line)
            if len(words) < 3:
                continue
            for i, _ in enumerate(words[:-1]):
                i = i + 1
                if words[i - 1] == words[i]:
                    if not all(x.isalpha() for x in words[i]):
                        continue

                    print(idx + 1, words[i], '\t', line)

# for i in filter(lambda x: x.split('.')[-1] == 'tex', d):
#    with open(reportFolder + i, 'r') as f:
#        print(i)
#        for idx, line in enumerate(f):
#            Warnings(idx + 1, line)
#        input()
