#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import sys
from feature_extractor import FeatureExtractor
import csv


random.seed = 7

dataFolder = sys.argv[1]
outfile = sys.argv[2]

csvfile = open(dataFolder)
authors = csv.reader(csvfile, delimiter=';')
next(authors)


N = []
N.extend(range(2, 11))
N.extend(range(12, 22, 2))
# N.extend(range(20, 40, 10))
# N.extend(range(40, 120, 20))

S = 500

POS = 219
SPEC = 524
CHAR = 2178
WORD = 188472
FREQ = 27535


# calls = [POS, SPEC, CHAR, WORD, FREQ]
#
# POS = SPEC = CHAR = WORD = FREQ = []
#
# for i in S:
#    for q in range(1, len(N)):
#        grams = combinations(N, q)
#
#        for combo in grams:
#            combo = list(combo)
#            postag_grams = list(map(lambda x: (x, i), combo))
#            special_grams = list(map(lambda x: (x, i), combo))
#            char_grams = list(map(lambda x: (x, i), combo))
#            word_grams = list(map(lambda x: (x, i), combo))
#            word_frequencies = i
#
#            POS.append(postag_grams)
#            SPEC.append(special_grams)
#            CHAR.append(char_grams)
#            WORD.append(word_grams)
#            FREQ.append(word_frequencies)
# print(char_grams)

POS = list(map(lambda x: (x, 50), N))
SPEC = list(map(lambda x: (x, 50), N))
CHAR = list(map(lambda x: (x, 300), N))
WORD = list(map(lambda x: (x, 500), N))
FREQ = list(map(lambda x: (x, 500), N))

feature_extractor = FeatureExtractor(authors,
                                     postag_grams=POS,
                                     special_character_grams=SPEC,
                                     word_grams=WORD,
                                     word_frequencies=500,
                                     character_grams=CHAR)

feature_extractor.extract(outfile)
csvfile.close()
