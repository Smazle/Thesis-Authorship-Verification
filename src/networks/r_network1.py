#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
import keras.layers as L


def model(reader):
    if not reader.binary:
        raise AttributeError('This network only works in binary mode.')

    known_in = L.Input(shape=(None, ), name='Known_Input')
    unknown_in = L.Input(shape=(None, ), name='Unknown_Input')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)
    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    if 'device:GPU' in str('str(device_lib.list_local_devices())'):
        gru = L.CuDNNGRU(200)
    else:
        gru = L.GRU(200)

    gru_known = gru(known_emb)
    gru_unknown = gru(unknown_emb)

    cos_distance = L.merge([gru_known, gru_unknown], mode='cos', dot_axes=1)
    cos_distance = L.Reshape((1, ))(cos_distance)
    cos_similarity = L.Lambda(lambda x: 1 - x)(cos_distance)

    model = Model(inputs=[known_in, unknown_in], outputs=cos_similarity)
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
