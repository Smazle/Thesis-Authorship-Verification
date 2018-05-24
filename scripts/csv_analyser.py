#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import jsonpickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/util/')
import utilities as util
from nltk.tokenize import sent_tokenize, word_tokenize


def fix_space(x):
    return x.replace(' ', '\\s')


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

parser.add_argument(
    '-j',
    '--json',
    help='Write stats to json file',
    type=bool,
    default=False,
    nargs='?'
)

args = parser.parse_args()

f = open(args.datafile, 'r', encoding='utf-8')
f.readline()  # Skip first line.


lower = 400
upper = 30000
sent = 500

lengths = []
authors = {}
charset = Counter()
uniques = []
words = []
sentences = []
sentence_count = []
uniqueWords = []

totalRemoved = 0

for line in f:
    author, time, text = line.split(';')
    text = util.clean(text)

    lengths.append(len(text))
    uniques.append(len(set(list(text))))

    tokens = sent_tokenize(text)
    sentence_count += [len(tokens)]
    sent_words = [len(util.wordProcess(x)) for x in tokens]

    sentences += [sum(sent_words) / len(tokens)] if len(tokens) != 0 else [0]

    cleanText = util.wordProcess(text)
    words.append(len(cleanText))
    uniqueWords.append(len(set(cleanText)))

    if author in authors:
        authors[author].append(len(text))
    else:
        authors[author] = [len(text)]

    if len(text) < lower or len(text) > upper:
        totalRemoved += 1
    elif sentence_count[-1] > sent:
        totalRemoved += 1

    charset = charset + Counter(text)

lengths = sorted(lengths)
charcount = float(sum(lengths))


stats = {}

stats['num_txt'] = len(lengths)

stats['avg_char_count'] = np.average(lengths)
stats['med_char_count'] = np.median(lengths)
stats['max_char_count'] = np.max(lengths)
stats['min_char_count'] = np.min(lengths)
stats['std_char_count'] = np.std(lengths)
stats['char_under_min'] = len(list(filter(lambda x: x < lower, lengths)))
stats['char_over_max'] = len(list(filter(lambda x: x > upper, lengths)))

stats['avg_char_unique'] = np.average(uniques)
stats['med_char_unique'] = np.median(uniques)
stats['max_char_unique'] = np.max(uniques)
stats['min_char_unique'] = np.min(uniques)
stats['std_char_unique'] = np.std(uniques)
stats['unique_char_over_max'] = len(list(filter(lambda x: x >= 100, uniques)))

stats['avg_word_count'] = np.average(words)
stats['med_word_count'] = np.median(words)
stats['max_word_count'] = np.max(words)
stats['min_word_count'] = np.min(words)
stats['std_word_count'] = np.std(words)

stats['avg_word_unique'] = np.average(uniqueWords)
stats['med_word_unique'] = np.median(uniqueWords)
stats['max_word_unique'] = np.max(uniqueWords)
stats['min_word_unique'] = np.min(uniqueWords)
stats['std_word_unique'] = np.std(uniqueWords)

stats['avg_sent_len'] = np.average(sentences)
stats['max_sent_len'] = np.max(sentences)
stats['min_sent_len'] = np.min(sentences)
stats['std_sent_len'] = np.std(sentences)

stats['avg_sent_count'] = np.average(sentence_count)
stats['max_sent_count'] = np.max(sentence_count)
stats['min_sent_count'] = np.min(sentence_count)
stats['std_sent_count'] = np.std(sentence_count)
stats['sent_count_over_max'] = len(
    list(filter(lambda x: x > sent, sentence_count)))

author_number_texts = list(map(lambda x: len(x), authors.values()))
stats['auth_num'] = len(authors)
stats['avg_auth_txt'] = np.average(author_number_texts)
stats['med_auth_txt'] = np.median(author_number_texts)
stats['max_auth_txt'] = np.max(author_number_texts)
stats['min_auth_txt'] = np.min(author_number_texts)
stats['std_auth_txt'] = np.std(author_number_texts)

if args.json:
    with open('stats.json', 'w') as f:
        f.write(jsonpickle.encode(stats))

print('\nNumber of texts', stats['num_txt'])

print('\nAverage character count', stats['avg_char_count'])
print('Median character count', stats['med_char_count'])
print('Max character count', stats['max_char_count'])
print('Min character count', stats['min_char_count'])
print('Std character count', stats['std_char_count'])
print('Texts Under Min %s characters' % lower, stats['char_under_min'])
print('Texts Over Max %s characters' % upper, stats['char_over_max'])

print('\nAverage Unique Characters', stats['avg_char_unique'])
print('Median Unique Characters', stats['med_char_unique'])
print('Min Unique Characters', stats['min_char_unique'])
print('Max Unique Characters', stats['max_char_unique'])
print('Std Unique Characters', stats['std_char_unique'])
print('Percieved Garbage Texts', stats['unique_char_over_max'])

print('\nAverage word count', stats['avg_word_count'])
print('Median word count', stats['med_word_count'])
print('Max word count', stats['max_word_count'])
print('Min word count', stats['min_word_count'])
print('Std word count', stats['std_word_count'])

print('\nAverage Unique Word', stats['avg_word_unique'])
print('Median Unique Word', stats['med_word_unique'])
print('Max Unique Word', stats['max_word_unique'])
print('Min Unique Word', stats['min_word_unique'])
print('Std Unique Word', stats['std_word_unique'])

print('\nAverage Sentence Length', stats['avg_sent_len'])
print('Max Sentence Length', stats['max_sent_len'])
print('Min Sentence Length', stats['min_sent_len'])
print('Std Sentence Length', stats['std_sent_len'])

print('\nAverage Sentence Count', stats['avg_sent_count'])
print('Max Sentence Count', stats['max_sent_count'])
print('Min Sentence Count', stats['min_sent_count'])
print('Std Sentence Count', stats['std_sent_count'])
print('Texts with over %s sentences' % sent, stats['sent_count_over_max'])

print('\nAuthor number', stats['auth_num'])
print('Author average text number', stats['avg_auth_txt'])
print('Author median text number', stats['med_auth_txt'])
print('Author max text number', stats['max_auth_txt'])
print('Author min text number', stats['min_auth_txt'])
print('Author std text number', stats['std_auth_txt'])

print('Total Removed ', totalRemoved)


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
