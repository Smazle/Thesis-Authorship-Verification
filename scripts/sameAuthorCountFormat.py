#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

file = sys.argv[1]
output = sys.argv[-1]

a = {}

with open(file) as f:
    next(f)
    for line in f:
        author = line.split(',')[0]
        if author in a:
            a[author] += 1
        else:
            a[author] = 1


minAuthorCount = min(a.values())
minAuthorCount = 6
print(len(list(filter(lambda x: x >= minAuthorCount, a.values()))))

b = {}

with open(file) as f, open(output, 'w') as out:
    out.write(next(f))
    for line in f:
        author = line.split(',')[0]
        if a[author] >= minAuthorCount:
            if author in b:
                b[author] += 1
            else:
                b[author] = 1

            if b[author] > minAuthorCount:
                continue
            else:
                out.write(line)


a = {}

with open(output) as f:
    next(f)
    for line in f:
        author = line.split(',')[0]
        if author in a:
            a[author] += 1
        else:
            a[author] = 1

print(a, minAuthorCount)
print(len(a.keys()))
