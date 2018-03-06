#!/usr/bin/python3

import argparse
import numpy as np
from collections import Counter

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
for line in f:
    author, text = line.split(';')

    lengths.append(len(text))

    if author in authors:
        authors[author].append(len(text))
    else:
        authors[author] = [len(text)]

    charset = charset + Counter(text)

lengths = sorted(lengths)

print('Number of texts', len(lengths))

print('Average length', np.average(lengths))
print('Median length', np.median(lengths))
print('Max length', np.max(lengths))
print('Min length', np.min(lengths))

author_number_texts = list(map(lambda x: len(x), authors.values()))
print('Author number', len(authors))
print('Author average text number', np.average(author_number_texts))
print('Author median text number', np.median(author_number_texts))
print('Author max text number', np.max(author_number_texts))
print('Author min text number', np.min(author_number_texts))

print('Charset', charset)
