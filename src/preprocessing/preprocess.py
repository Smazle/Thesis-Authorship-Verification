#!/usr/bin/env python3 # -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import argparse
import platform
from nltk.tokenize import sent_tokenize

from src.util import utilities as util


parser = argparse.ArgumentParser(
    'Applies preprocessing to specified file'
)

parser.add_argument(
    'filename',
    type=str,
    help='Path to file, designated for preprocessing'
)

parser.add_argument(
    '-rc',
    '--remove-char',
    type=int,
    help='Removed the first characters in each text, \
                the amount of which is specified by this argument. \
                Keep in mind that this is done before anything else'
)

parser.add_argument(
    '-ucl',
    '--upper-char-limit',
    type=int,
    help='The upper limit of the amount \
                of characters accepted in each text'
)

parser.add_argument(
    '-lcl',
    '--lower-char-limit',
    type=int,
    help='The lower limit of the amount \
                of characters accpted in each text'
)

parser.add_argument(
    '-usl',
    '--upper-sentence-limit',
    type=int,
    help='The upper limit on how many \
                sentences are accepted in each text'
)

args = parser.parse_args()

under = []
over = []
sents = []

dataFile = pd.read_csv(args.filename, sep=';', encoding='utf-8')
print(dataFile.columns)

c = 0

for idx, row in dataFile.iterrows():
    if idx % round(0.1 * len(dataFile)) == 0:
        print(str(10 * c) + '%')
        c += 1

    text = row['Text']
    text = util.clean(text)
    # Remove starting characters
    text = str(text[args.remove_char:])

    dataFile['Text'][idx] = text

    sent = sent_tokenize(text)
    l_txt = len(text)

    if l_txt > args.upper_char_limit:
        over.append(idx)
    elif l_txt < args.lower_char_limit:
        under.append(idx)
    elif len(sent) > args.upper_sentence_limit:
        sents.append(idx)


print(len(under), len(over), len(sents), len(under) + len(over) + len(sents))
dataFile = dataFile.drop(under + over + sents)


splitChar = '/' if platform.system() == 'Linux' else '\\'

name = args.filename.split('\\')

f = name[-1].split('.')
f.insert(1, '_processed.')
name[-1] = ''.join(f)

dataFile.index = range(len(dataFile))

dataFile.to_csv(splitChar.join(name), sep=';', index=False)
