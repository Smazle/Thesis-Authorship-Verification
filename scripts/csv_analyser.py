#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ..src.util import util
from nltk.tokenize import sent_tokenize, word_tokenize


def fix_space(x):
    return x.replace(' ', '{SPACE}')


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
words = []
sentences = []
sentence_count = []
uniqueWords = []
for line in f:
    author, time, text = line.split(';')
    text = util.clean(text)

    lengths.append(len(text))
    uniques.append(len(set(list(text))))

    tokens = sent_tokenize(text)
    sentence_count += [len(tokens)]
    words = [len(util.wordProcess(x)) for x in tokens]
    sentences += words

    cleanText = util.wordProcess(text)
    words.append(len(cleanText))
    uniqueWords.append(len(set(cleanText)))

    if author in authors:
        authors[author].append(len(text))
    else:
        authors[author] = [len(text)]

    charset = charset + Counter(text)

lengths = sorted(lengths)
charcount = float(sum(lengths))

print('\nNumber of texts', len(lengths))

print('\nAverage character count', np.average(lengths))
print('Median character count', np.median(lengths))
print('Max character count', np.max(lengths))
print('Min character count', np.min(lengths))
print('Texts Under Min characters', len(
    list(filter(lambda x: x < 400, lengths))))
print('Texts Over Max characters', len(
    list(filter(lambda x: x > 30000, lengths))))

print('\nAverage Unique Characters', np.average(uniques))
print('Median Unique Characters', np.median(uniques))
print('Min Unique Characters', np.min(uniques))
print('Max Unique Characters', np.max(uniques))
print('Percieved Garbage Texts',
      len(list(filter(lambda x: x >= 100, uniques))))

print('\nAverage word count', np.average(words))
print('Median word count', np.median(words))
print('Max word count', np.max(words))
print('Min word count', np.min(words))

print('\nAverage Unique Characters', np.average(uniqueWords))
print('Median Unique Characters', np.median(uniqueWords))
print('Min Unique Characters', np.min(uniqueWords))
print('Max Unique Characters', np.max(uniqueWords))

print('\nAverage Sentence Length', np.average(sentences))
print('Median Sentence Length', np.median(sentences))
print('Min Sentence Length', np.min(sentences))
print('Max Sentence Length', np.max(sentences))

print('\nAverage Sentence Count', np.average(sentence_count))
print('Median Sentence Count', np.median(sentence_count))
print('Min Sentence Count', np.min(sentence_count))
print('Max Sentence Count', np.max(sentence_count))

author_number_texts = list(map(lambda x: len(x), authors.values()))
print('\nAuthor number', len(authors))
print('Author average text number', np.average(author_number_texts))
print('Author median text number', np.median(author_number_texts))
print('Author max text number', np.max(author_number_texts))
print('Author min text number', np.min(author_number_texts))


# æøå freq
d_char = ['æ', 'ø', 'å']
d_freq = [(charset[x] / charcount, charset[x.upper()] / charcount)
          for x in d_char]

print()
for i, char in enumerate(d_freq):
    print('Frequencies, %s %s, %s %s' %
          (d_char[i], char[0], d_char[i].upper(), char[1]))


# Character Destribution
charset = sorted(charset.items(), key=lambda x: x[1], reverse=True)

characters = list(map(lambda x: fix_space(x[0]), charset))
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
