#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(
    description='Splits a given featurefile into\
                training and validation'
)


parser.add_argument(
    'featurefile',
    type=str,
    help='Path to the featurefile'
)


parser.add_argument(
    'ratio',
    type=float,
    help='The ratio of the training set which should\
                be used for training'
)


args = parser.parse_args()


print('Reading Files')
with open(args.featurefile, 'r') as featurefile:
    data = pd.read_csv(featurefile)
    authors = data.as_matrix(columns=['author']).flatten()


print('Shuffling Authors')
unique_authors = np.unique(authors)
np.random.shuffle(unique_authors)


print('Splitting Authors')
split = int(args.ratio * len(unique_authors))
trainingAuthors = unique_authors[:split]
validationAuthors = unique_authors[split:]

dataMat = data.as_matrix()
trainingData = dataMat[np.isin(authors, trainingAuthors)]
validationData = dataMat[np.isin(authors, validationAuthors)]


trainingData = pd.DataFrame(data=trainingData,
                            columns=list(data))

validationData = pd.DataFrame(data=validationData,
                              columns=list(data))

fName = args.featurefile.split('.')

print('Writing to files {}.(training/validation)'.format(args.featurefile))
trainingData.to_csv(args.featurefile + '.training')
validationData.to_csv(args.featurefile + '.validation')
