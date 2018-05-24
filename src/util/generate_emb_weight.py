#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def GetEmbeddingWeights(emb_path: str, reader):

    weights = []
    v_map = reader.channels[0].vocabulary_map

    with open(emb_path, 'r', encoding='utf-8') as f:
        X, y = [int(x.strip()) for x in next(f).split(' ')]

        weights = np.zeros(
            (len(reader.channels[0].vocabulary_above_cutoff) + 2, y))

        for idx, line in enumerate(f):
            line = line.split(' ')
            word = line[0]

            w = [float(x) for x in line[1:-1]]

            if word in v_map:
                weights[v_map[word], :] = w

    return weights
