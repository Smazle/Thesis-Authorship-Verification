#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

WORD_VEC_PATH = os.path.join('.', 'data', 'pre-trained', 'wiki.da.vec')


def generate_embedding_weights(vocabulary_map, emb_path=WORD_VEC_PATH):

    word_n = len(vocabulary_map)

    with open(emb_path, 'r', encoding='utf-8') as f:
        X, y = [int(x.strip()) for x in next(f).split(' ')]

        weights = np.zeros((word_n + 2, y))

        for idx, line in enumerate(f):
            line = line.split(' ')
            word = line[0]

            w = [float(x) for x in line[1:-1]]

            if word in vocabulary_map:
                weights[vocabulary_map[word], :] = w

    return weights
