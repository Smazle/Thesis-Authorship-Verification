#!/usr/bin/python3
# -*- coding: utf-8 -*-

import keras.layers as L
from keras.models import Model


def model(reader):
    inshape = (reader.max_len, )

    known_in = L.Input(shape=inshape, name='known_input')
    unknown_in = L.Input(shape=inshape, name='unknown_input')

    embedding = L.Embedding(
        len(reader.vocabulary_above_cutoff) + 2,
        5,
        input_length=reader.max_len)

    conv = L.Convolution1D(
        filters=1000,
        kernel_size=10,
        strides=1,
        activation='relu',
        name='convolution_10')

    repr_known = L.GlobalMaxPooling1D(name='repr_known')(conv(
        embedding(known_in)))
    repr_unknown = L.GlobalMaxPooling1D(name='repr_unknown')(conv(
        embedding(unknown_in)))

    full_input = L.Concatenate()([repr_known, repr_unknown])

    dense = L.Dense(500, activation='relu')(full_input)
    output = L.Dense(2, activation='softmax', name='output')(dense)

    model = Model(inputs=[known_in, unknown_in], outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
