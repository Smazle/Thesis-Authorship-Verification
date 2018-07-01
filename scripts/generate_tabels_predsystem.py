# -*- coding: utf-8 -*-
import sys

mapper = {
    'MajorityVote': '$P_{mv}$',
    'Minimum': '$P_{min}$',
    'Maximum': '$P_{max}$',
    'TextLength': '$P_{l}$',
    'Expλ=1.0': '$P_{exp_{1.0}}$',
    'Expλ=0.75': '$P_{exp_{0.75}}$',
    'Expλ=0.5': '$P_{exp_{0.5}}$',
    'Expλ=0.25': '$P_{exp_{0.25}}$',
    'Expλ=0.25+': '$P_{lexp_{0.25}}$',
    'Expλ=0.0': '$P_{U}$'
}

max_acc = []
limit = []

with open(sys.argv[-1], 'r') as f:
    next(f)
    for line in f:
        line = line.strip().split(' ')
        if len(line) > 1:
            line = list(filter(lambda x: not x == '', line))

            temp = []
            for idx in range(1, len(line)):
                try:
                    float(line[idx])
                except ValueError:
                    line[0] += line[idx]
                    temp += [idx]

            for i in temp:
                del line[1]

            w, a_err, theta, acc, acc_err, tps, tns, fps, fns = line

            line = [w, mapper[w], theta, tps, tns, fps, fns, acc, acc_err]
            line = '\t'.join(line)

            max_acc.append((acc, line))
            if a_err == '0.1':
                limit.append((int(tns), line))

for i in sorted(limit, key=lambda x: x[0], reverse=True):
    print(i[1])

for i in sorted(limit, key=lambda x: x[0], reverse=True):
    print(i[1])
