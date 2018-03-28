#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import sys

# Read datafile without consuming all my RAM (numpy...).
with open(sys.argv[1], 'r') as feature_file:
    feature_file.readline()  # Skip first line.
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    # Number of features is number of columns minus the author column.
    feature_n = len(reader.__next__()) - 1

    line_n = 1
    for line in reader:
        line_n = line_n + 1

    X = np.zeros((line_n, feature_n), dtype=np.float)
    authors = np.zeros((line_n, ), dtype=np.int)

    # Go back to start of file and read again.
    feature_file.seek(0)
    feature_file.readline()
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    for i, line in enumerate(reader):
        X[i] = np.array(list(map(lambda x: float(x), line[0:-1])))
        authors[i] = int(line[-1])
