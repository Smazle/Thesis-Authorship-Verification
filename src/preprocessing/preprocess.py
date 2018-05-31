#!/usr/bin/env python3
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

f = open(args.filename, 'r', encoding='utf-8')
header = next(f)

c = 0

splitChar = '/' if platform.system() == 'Linux' else '\\'

name = args.filename.split('\\')

out = name[-1].split('.')
out.insert(1, '_processed.')
name[-1] = ''.join(out)

with open(splitChar.join(name), 'w') as output:
    output.write(header)

    for idx, line in enumerate(f):
        line = line.strip()
        author, date, text = line.split(';')

        if idx % 1000 == 0:
            print(idx)

        raw_text = text[args.remove_char:]

        text = util.clean(raw_text)

        # Remove starting characters
        sent = sent_tokenize(text)
        l_txt = len(text)

        if l_txt > args.upper_char_limit:
            over.append(idx)
            continue
        elif l_txt < args.lower_char_limit:
            under.append(idx)
            continue
        elif len(sent) > args.upper_sentence_limit:
            sents.append(idx)
            continue

        output.write(';'.join([author, date, raw_text]))

print(len(under), len(over), len(sents), len(under) + len(over) + len(sents))
