#!/usr/bin/python3
# -*- coding: utf-8 -*-

import keras.layers as L
import keras.optimizers as O
from keras.models import Model


def model(reader):
    inshape = (None, )

    known_in = L.Input(shape=inshape, name='known_input', dtype='int32')
    unknown_in = L.Input(shape=inshape, name='unknown_input', dtype='int32')

    embedding = L.Embedding(
        len(reader.channels[0].vocabulary_above_cutoff) + 2, 5)

    known_embed = embedding(known_in)
    unknown_embed = embedding(unknown_in)

    conv8 = L.Convolution1D(
        filters=5000,
        kernel_size=8,
        strides=1,
        activation='relu',
        name='convolutional_8')

    repr_known = L.GlobalMaxPooling1D()(conv8(known_embed))
    repr_unknown = L.GlobalMaxPooling1D()(conv8(unknown_embed))

    abs_diff = L.merge(
        inputs=[repr_known, repr_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference')

    dense1 = L.Dense(500, activation='relu')(abs_diff)
    dense2 = L.Dense(500, activation='relu')(dense1)
    dense3 = L.Dense(500, activation='relu')(dense2)
    dense4 = L.Dense(500, activation='relu')(dense3)
    dense5 = L.Dense(500, activation='relu')(dense4)

    pruned = L.Dropout(0.3)(dense5)

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    optimizer = O.Adam(lr=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
