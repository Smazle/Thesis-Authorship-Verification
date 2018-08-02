#!/usr/bin/python3

import numpy as np
import pandas as pd
import sys

data = pd.read_csv(sys.stdin, delimiter=',', quotechar='"', escapechar='\\')

filters = data.as_matrix(columns=['filter']).flatten()
activation_strings = data.as_matrix(columns=['activation_string']).flatten()
activation_values = data.as_matrix(columns=['activation_value']).flatten()

for f in np.sort(np.unique(filters)):
    print(f, file=sys.stderr)
    values = activation_values[filters == f]
    strings = activation_strings[filters == f]

    sort = np.argsort(values)
    largest_strings = strings[sort][-3:]
    print(largest_strings)
