#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


parse = argparse.ArgumentParser(
    'Splits the provided data into chunks based on the \
                given parameters'
)

parse.add_argument(
    'datafile',
    type=str,
    help='Path to the data file from which to extract a certain amount \
                of authors'
)
parse.add_argument(
    'outfile',
    type=str,
    help='Path to miniturized output file'
)

parse.add_argument(
    '--extract',
    help='How many authors to extract, float for percentage, int for count'
)

args = parse.parse_args()

data = pd.read_csv(args.datafile, delimiter=';')
authors = data.as_matrix(columns=['StudentId']).flatten()
unique_authors = np.unique(authors)

try:
    split = int(args.extract)
    unique_authors = np.random.choice(unique_authors, split, replace=False)
except ValueError:
    split = float(args.extract)
    unique_authors = np.random.choice(
        unique_authors, int(len(authors) * split), replace=False)

idx = np.isin(authors, unique_authors)
idx = np.where(idx)[0]
output = data.iloc[idx]
output.index = range(len(output))

print(output.shape)
output.to_csv(args.outfile, sep=';', encoding='utf-8', index=False)
