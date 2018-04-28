#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
import keras.layers as L


def model(reader):
    known_in = L.Input(shape=(None, ), dtype='int32')
    unknown_in = L.Input(shape=(None, ), dtype='int32')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    if 'device:GPU' in str('str(device_lib.list_local_devices())'):
        gru = L.CuDNNGRU(200)
    else:
        gru = L.GRU(200)

    features_known = gru(known_emb)
    features_unknown = gru(unknown_emb)

    abs_diff = L.merge(
        inputs=[features_known, features_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    dense1 = L.Dense(500, activation='relu')(abs_diff)
    dense2 = L.Dense(500, activation='relu')(dense1)

    pruned = L.Dropout(0.3)(dense1)

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model
