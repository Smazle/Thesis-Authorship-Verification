#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
import keras.layers as L


def model(reader):
    inshape = (None, )

    known_in = L.Input(shape=inshape, dtype='int32')
    unknown_in = L.Input(shape=inshape, dtype='int32')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    conv8 = L.Convolution1D(filters=500, kernel_size=8, strides=1,
                            activation='relu', padding='same')
    pool = L.MaxPooling1D(pool_size=8)

    if 'device:GPU' in str('str(device_lib.list_local_devices())'):
        gru = L.CuDNNGRU(100)
    else:
        gru = L.GRU(200)

    features_known = gru(pool(conv8(known_emb)))
    features_unknown = gru(pool(conv8(unknown_emb)))

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
