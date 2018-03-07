#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '../src/util/')
import utilities as util

parser = argparse.ArgumentParser(
    description=(
        'Return statistics about the CSV files we get from MaCom. ' +
        'The statistics is supposed to be included as part of the report'
    )
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to CSV file'
)
args = parser.parse_args()

f = open(args.datafile, 'r', encoding='utf-8')
f.readline()  # Skip first line.

lengths = []
authors = {}
charset = Counter()
uniques = []
for line in f:
    author, text = line.split(';')
    text = util.clean(text)

    lengths.append(len(text))
    uniques.append(len(set(list(text))))

    if author in authors:
        authors[author].append(len(text))
    else:
        authors[author] = [len(text)]

    charset = charset + Counter(text)

lengths = sorted(lengths)
charcount = float(sum(lengths))

print('\nNumber of texts', len(lengths))

print('\nAverage length', np.average(lengths))
print('Median length', np.median(lengths))
print('Max length', np.max(lengths))
print('Min length', np.min(lengths))

print('\nAverage Unique Characters', np.average(uniques))
print('Median Unique Characters', np.median(uniques))
print('Min Unique Characters', np.min(uniques))
print('Max Unique Characters', np.max(uniques))
print('Percieved Garbage Texts',
      len(list(filter(lambda x: x >= 100, uniques))))

author_number_texts = list(map(lambda x: len(x), authors.values()))
print('\nAuthor number', len(authors))
print('Author average text number', np.average(author_number_texts))
print('Author median text number', np.median(author_number_texts))
print('Author max text number', np.max(author_number_texts))
print('Author min text number', np.min(author_number_texts))


d_char = ['æ', 'ø', 'å']
d_freq = [(charset[x] / charcount, charset[x.upper()] / charcount)
          for x in d_char]

print()
for i, char in enumerate(d_freq):
    print('Frequencies, %s %s, %s %s' %
          (d_char[i], char[0], d_char[i].upper(), char[1]))

charset = sorted(charset.items(), key=lambda x: x[1], reverse=True)

characters = list(map(lambda x: x[0], charset))
frequencies = list(map(lambda x: x[1] / charcount, charset))
X = range(len(characters))


plt.title('Character Frequencies and Threshold')
plt.xlabel('Character')
plt.ylabel('Frequency')

ticks = [int(0 + (len(X) * 0.05) * x)
         for x in range(int(len(X) / (len(X) * 0.05)))] + [len(X) - 1]
characters = np.array(characters)
plt.xticks(ticks, characters[ticks])


line1 = plt.semilogy(X, frequencies, color='blue', label='Frequency')
line2 = plt.plot(X, [1.0 / 100000.0] * len(X), color='red', label='Threshold')
plt.legend(handles=[line1[0], line2[0]])

plt.savefig('Frequencies.png')

# print('Charset', charset)