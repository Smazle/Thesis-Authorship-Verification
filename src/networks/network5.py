#!/usr/bin/python3
# -*- coding: utf-8 -*-

import keras.layers as L
from keras.models import Model


def model(reader):
    inshape = (None, )

    known_in = L.Input(shape=inshape, name='known_input')
    unknown_in = L.Input(shape=inshape, name='unknown_input')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5,
                            input_length=reader.max_len)

    known_embed = embedding(known_in)
    unknown_embed = embedding(unknown_in)

    conv8 = L.Convolution1D(filters=500, kernel_size=8, strides=1,
                            activation='relu', name='convolutional_8')

    conv4 = L.Convolution1D(filters=500, kernel_size=4, strides=1,
                            activation='relu', name='convolutional_4')

    conv16 = L.Convolution1D(filters=200, kernel_size=8, strides=1,
                             activation='relu', name='convolutional_8_2')

    repr_known1 = L.GlobalMaxPooling1D(name='known_repr_8')(
        conv8(known_embed))
    repr_unknown1 = L.GlobalMaxPooling1D(name='unknown_repr_8')(
        conv8(unknown_embed))

    repr_known2 = L.GlobalMaxPooling1D(name='known_repr_4')(
        conv4(known_embed))
    repr_unknown2 = L.GlobalMaxPooling1D(name='unknown_repr_4')(
        conv4(unknown_embed))

    repr_known3 = L.GlobalMaxPooling1D(name='known_repr_8_2')(
        conv16(known_embed))
    repr_unknown3 = L.GlobalMaxPooling1D(name='unknown_repr_8_2')(
        conv16(unknown_embed))

    repr_known = L.Concatenate()([repr_known1, repr_known2, repr_known3])
    repr_unknown = L.Concatenate()(
        [repr_unknown1, repr_unknown2, repr_unknown3])

    abs_diff = L.merge(
        inputs=[repr_known, repr_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    dense1 = L.Dense(500, activation='relu')(abs_diff)
    pruned = L.Dropout(0.3)(dense1)
    dense2 = L.Dense(500, activation='relu')(pruned)
    dense3 = L.Dense(500, activation='relu')(dense2)
    dense4 = L.Dense(500, activation='relu')(dense3)

    output = L.Dense(2, activation='softmax', name='output')(dense4)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
