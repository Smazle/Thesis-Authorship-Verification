#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from .feature_extractor import FeatureExtractor
import argparse
import csv
import sys


# Make sure we can hold all files in csv file memory.
csv.field_size_limit(2147483647)

# Set random state for reproducible results.
random.seed = 7

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Run feature extraction on a datafile in MaCom CSV format.'
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to file containing texts in csv format.'
)
parser.add_argument(
    'outfile',
    type=str,
    help='Path to file we should write output in.'
)
parser.add_argument(
    '--skip-lines',
    type=int,
    default=1,
    help='How many lines to skip in the beginning of the datafile.'
)
args = parser.parse_args()

with open(args.datafile, 'r', encoding='utf-8') as csvfile:
    authors = csv.reader(csvfile, delimiter=';')

    # If we are asked to skip lines we will.
    for i in range(args.skip_lines):
        next(authors)

    postag_grams = list(map(lambda x: (x, 50), [3, 4]))
    special_character_grams = list(map(lambda x: (x, 50), [2, 3, 4]))
    character_grams = list(map(lambda x: (x, 300), [2, 3, 4, 5, 6, 7, 8, 9, 10]))
    word_grams = list(map(lambda x: (x, 500), [2, 3, 4]))

    feature_extractor = FeatureExtractor(
        authors,
        postag_grams=postag_grams,
        special_character_grams=special_character_grams,
        word_grams=word_grams,
        word_frequencies=500,
        character_grams=character_grams,
    )

    feature_extractor.extract(args.outfile)
