#!/usr/bin/env python3

import sys
import count
import os
import numpy as np
import random

SEED = 7
NEWLINE = "$NL$"
SEMI = "$SC$"
PATH ="/home/smazle/Desktop/danske.csv"
MAX_LENGTH = 0

random.seed(SEED)


def Fix(text):
    # Replace escapes
    data = list(map(lambda x: 
        x.replace(NEWLINE, "\n").replace(SEMI, "\n"), text))
    return np.array(data)


student, text = np.loadtxt(PATH, delimiter=";", skiprows=1, dtype=str, unpack = True)
ids = list(map(str, range(0, len(student))))


data = np.c_[ids, student, Fix(text)]
data = np.array(sorted(data, key=lambda x: int(x[1])))





MAX_LENGTH = max([len(x[-1]) for x in data])

print(MAX_LENGTH)


class Author:
    def __init__(self, texts):
        self.texts = texts
        self.ordTexts = np.array([[ord(y) for y in x] for x in texts])

    def oneHotEncode(self, vocab):
        # Create oneHot encoding map, and add one for
        # padding purposes
        self.oneHotMap = np.diag(np.ones(len(vocab) + 1))

        self.encoded = np.array([[self.oneHotMap[vocab.index(x)] for x in y] for y in self.ordTexts])

    def pad(self, max_length):
        temp = []
        padValue = self.oneHotMap[-1,:]

        for i, item in enumerate(self.encoded):
            item = list(item)
            diff = max_length - len(item)

            if diff != 0:
                temp.append(list(np.concatenate((item, ([padValue] * diff))).shape))

            self.encoded = np.array(temp)



    def randomText(self):
        return random.choice(self.encoded) 





def GetProblems(authors, n):

    problems = []

    for _ in range(0, n):
        keys = list(authors.keys())
        randomKey = random.choice(keys)
        keys.remove(randomKey)

        author1 = authors[randomKey]
        author2 = authors[random.choice(keys)]

        same_sample = np.concatenate((author1.randomText(), author1.randomText(), [1]))
        different_sample = np.concatenate((author1.randomText(), author2.randomText(), [0]))

        problems.append(same_sample)
        problems.append(different_sample)

    return np.array(problems)







authors = {}

for i in range(3):
    authors[i] = Author(data[:,-1])


#for x in data:
#    student = x[1]
#    if student in authors:
#        authors[student].texts.append(x[-1])
#    else:
#        authors[student] = Author([x[-1]])

master_text = np.concatenate([[x[-1]] for x in data])
master_text = "".join(master_text)
master_text = [ord(x) for x in master_text]
vocabulary = list(set(master_text))

for _, author in authors.items():
    author.oneHotEncode(vocabulary)
    author.pad(MAX_LENGTH)

temp = GetProblems(authors, 1)
