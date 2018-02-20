#!/usr/bin/env python3

import numpy as np
import random

SEED = 7
NEWLINE = '$NL$'
SEMI = '$SC$'
PATH = '/home/smazle/Desktop/danske.csv'
MAX_LENGTH = 0

random.seed(SEED)


def Fix(text):
    # Replace escapes
    data = list(map(lambda x:
                    x.replace(NEWLINE, '\n').replace(SEMI, '\n'), text))
    return np.array(data)


student, text = np.loadtxt(
    PATH, delimiter=';', skiprows=1, dtype=str, unpack=True)
ids = list(map(str, range(0, len(student))))


data = np.c_[ids, student, Fix(text)]
data = np.array(sorted(data, key=lambda x: int(x[1])))

MAX_LENGTH = max([len(x[-1]) for x in data])


class Author:
    def __init__(self, texts):
        self.texts = texts
        self.ordTexts = np.array([[ord(y) for y in x] for x in texts])

    def oneHotEncode(self, vocab):
        # Create oneHot encoding map, and add one for
        # padding purposes
        self.oneHotMap = np.diag(np.ones(len(vocab) + 1))

        self.encoded = np.array(
            [[self.oneHotMap[vocab.index(x)] for x in y]
                for y in self.ordTexts])

    def pad(self, max_length):
        temp = []
        padValue = self.oneHotMap[-1, :]

        for i, item in enumerate(self.encoded):
            item = list(item)
            diff = max_length - len(item)

            if diff != 0:
                temp.append(list(np.concatenate((item, ([padValue] * diff)))))
            else:
                temp.append(item)

        self.encoded = np.array(temp)

    def randomText(self):
        return random.choice(self.encoded)


def GetProblems(authors, n):
    X = []
    y = []

    for _ in range(0, n):
        keys = list(authors.keys())
        randomKey = random.choice(keys)
        keys.remove(randomKey)

        author1 = authors[randomKey]
        author2 = authors[random.choice(keys)]

        print(author1.randomText().shape)
        print(author2.randomText().shape)
        print(author2.encoded.shape)
        print(author2.ordTexts.shape)

        print()

        same_sample = np.concatenate(
            (author1.randomText(), author1.randomText()))
        different_sample = np.concatenate(
            (author1.randomText(), author2.randomText()))

        X.append(same_sample)
        X.append(different_sample)
        y.append(1)
        y.append(0)

    return np.array(X), np.array(y)


def genVocabulary(data):
    master_text = ''.join(data)
    master_text = [ord(x) for x in master_text]
    vocabulary = list(set(master_text))
    return vocabulary


authors = {}

for x in data:
    student = x[1]
    if student in authors:
        authors[student].texts.append(x[-1])
    else:
        authors[student] = Author([x[-1]])

vocabulary = genVocabulary(data[:, -1])

for q, author in authors.items():
    author.oneHotEncode(vocabulary)
    author.pad(MAX_LENGTH)

X, y = GetProblems(authors, 1)
print(y)
