#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
import keras.layers as L


def model(reader):
    inshape = (reader.max_len, )

    known_in = L.Input(shape=inshape, dtype='int32')
    unknown_in = L.Input(shape=inshape, dtype='int32')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    lstm = L.Bidirectional(L.LSTM(100, return_sequences=True))

    char_features_known = lstm(known_emb)
    char_features_unknown = lstm(unknown_emb)

    features_known = L.GlobalAveragePooling1D()(char_features_known)
    features_unknown = L.GlobalAveragePooling1D()(char_features_unknown)

    cos_distance = L.merge(
        [features_known, features_unknown], mode='cos', dot_axes=1)
    cos_distance = L.Reshape((1, ))(cos_distance)
    cos_similarity = L.Lambda(lambda x: 1 - x)(cos_distance)

    model = Model(inputs=[known_in, unknown_in], outputs=cos_similarity)

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
