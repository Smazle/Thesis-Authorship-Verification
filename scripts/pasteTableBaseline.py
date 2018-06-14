# -*- coding: utf-8 -*-
import sys

temp = []

with open(sys.argv[-1], 'r') as f:
    for string in f:
        val = list(map(lambda x: x.strip('() \n'), string.split(',')))
        temp.append(val)

del temp[-1]

temp = sorted(temp, key=lambda x: float(x[0]))
print(temp)
val = []
for idx, w in enumerate(temp):
    if idx != 0 and temp[idx - 1][0] != w[0]:
        val = list(map(lambda x: x[:5], val))
        print(' & '.join(val))
        val = []
    val.append(w[-1])
val = list(map(lambda x: x[:5], val))
print(' & '.join(val))
