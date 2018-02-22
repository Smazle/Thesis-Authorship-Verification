#!/usr/bin/env python3

import sys
import numpy as np

args = sys.argv[1:]

dataFile = args[0]

data = np.loadtxt(dataFile)
data = np.array(sorted(data, key=lambda x: int(x[-1])))

print(len(data))

# Clear lonely authors
data = np.array([x for i, x in enumerate(data) if data[i-1]
                 [-1] == x[-1] or data[i+1][-1] == x[-1]])

print(len(data))

uID = np.array(data[:, -1], dtype=int)
data = data[:, :-1]
