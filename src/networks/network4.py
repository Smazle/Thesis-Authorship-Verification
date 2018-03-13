# -*- coding: utf-8 -*-
# !/usr/bin/python3
# -*- coding: utf-8 -*-

# from keras.preprocessing import sequence
from ..preprocessing import MacomReader
import numpy as np
import argparse

np.random.seed(7)

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to data file.'
)
parser.add_argument(
    '--history',
    type=str,
    help='Path to file to write history to.',
    default=None
)
parser.add_argument(
    '--graph',
    type=str,
    help='Path to file to visualize network in.'
)
args = parser.parse_args()


reader = MacomReader(
    args.datafile,
    batch_size=8,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000
)
