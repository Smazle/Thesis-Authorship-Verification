#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..preprocessing.channels import ChannelType
from keras.models import Model
import keras.layers as L
import keras.optimizers as O


def model(reader):
    assert reader.channeltypes == [ChannelType.POS_TAGS]

    known_in = L.Input(shape=(None, ), dtype='int32')
    unknown_in = L.Input(shape=(None, ), dtype='int32')

    embedding = L.Embedding(
        len(reader.channels[0].vocabulary_above_cutoff) + 2, 1)

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    feature_extractor1 = L.Bidirectional(L.LSTM(100, return_sequences=True))

    pool = L.GlobalAvgPool1D()

    features_known = pool(feature_extractor1(known_emb))
    features_unknown = pool(feature_extractor1(unknown_emb))

    abs_diff = L.merge(
        inputs=[features_known, features_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference')

    # Dense network.
    dense1 = L.Dense(300)(abs_diff)
    dense2 = L.Dense(200)(dense1)
    dense3 = L.Dense(100)(dense2)
    output1 = L.Dense(2, activation='softmax')(dense1)
    output2 = L.Dense(2, activation='softmax')(dense2)
    output3 = L.Dense(2, activation='softmax')(dense3)
    output = L.Average()([output1, output2, output3])

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    optimizer = O.Adam(lr=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
