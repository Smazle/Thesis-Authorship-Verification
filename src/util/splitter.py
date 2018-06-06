#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    'Splits the provided data into two pieces based on the \
     given parameters.')
parser.add_argument(
    'datafile',
    type=str,
    help='Path to the data file from which to extract a certain amount \
          of authors.')
parser.add_argument(
    'outfile', type=str, help='Path to output file of given size.')
parser.add_argument(
    'rest',
    type=str,
    help='Path to file where the rest of the authors are written.')
parser.add_argument(
    '--extract', help='How many authors to extract into outfile.', type=int)
args = parser.parse_args()

data = pd.read_csv(args.datafile, delimiter=';')
authors = data.as_matrix(columns=['StudentId']).flatten()
unique_authors = np.unique(authors)

split = int(args.extract)
chosen_authors = np.random.choice(unique_authors, split, replace=False)

outfile_content = data.iloc[np.where(np.isin(authors, chosen_authors))]
rest = data.iloc[np.where(np.logical_not(np.isin(authors, chosen_authors)))]

outfile_content.index = range(len(outfile_content))
rest.index = range(len(rest))

outfile_content.to_csv(args.outfile, sep=';', encoding='utf-8', index=False)
rest.to_csv(args.rest, sep=';', encoding='utf-8', index=False)
