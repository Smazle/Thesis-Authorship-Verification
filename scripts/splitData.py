#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
# from collections import Counter


parser = argparse.ArgumentParser(
    description=(
        'Splits the selected file into training, validation' +
        ' and test set, with each author being represented' +
        'in all sets'
    )
)


parser.add_argument(
    'datafile',
    type=str,
    help='Path to CSV file'
)
parser.add_argument(
    '--split',
    type=list,
    help='Training Validation and Test split',
    nargs='+'
)

args = parser.parse_args()
splits = [float(''.join(x)) for x in args.split]


split = {'train': [], 'val': [], 'test': []}

authorID = subprocess.check_output(
    "cut -d\";\" -f1 " + args.datafile, shell=True).decode('utf-8')
authorID = authorID.split('\n')
del authorID[0]
del authorID[-1]

authorID = list(map(int, authorID))
uniqueAuthors = list(set(authorID))

count = len(authorID)

if any(list(map(lambda x: (x * count) < len(uniqueAuthors), split))):
    raise Exception("Split doesn't allow for all author to be represented")

for i in uniqueAuthors:
    selection = list(filter(lambda x: x[1] == i, authorID))

    split['val'].append(selection[0][0])
    del authorID[selection[0][0]]
    split['test'].append(selection[1][0])
    del authorID[selection[1][0]]

diffVal = int(count * split[1]) - len(split['val'])
diffTest = int(count * split[2]) - len(split['test'])


# print(authorID)
