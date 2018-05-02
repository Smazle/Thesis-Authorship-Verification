#!/usr/bin/python3
# -*- coding: utf-8 -*-

import keras.layers as L
from keras.models import Model


def model(reader):
    inshape = (None, )

    known_in = L.Input(shape=inshape, name='known_input')
    unknown_in = L.Input(shape=inshape, name='unknown_input')

    embedding = L.Embedding(len(reader.vocabulary_above_cutoff) + 2, 5,
                            input_length=reader.max_len, name='embedding')

    known_embed = embedding(known_in)
    unknown_embed = embedding(unknown_in)

    # Convolution 1

    conv8_1 = L.Convolution1D(filters=500, kernel_size=8, strides=1,
                              activation='relu', name='convolutional_8_1')

    conv4_1 = L.Convolution1D(filters=500, kernel_size=4, strides=1,
                              activation='relu', name='convolutional_4_1')

    conv16_1 = L.Convolution1D(filters=500, kernel_size=16, strides=1,
                               activation='relu', name='convolutional_16_1')

    max_1_1 = L.MaxPooling1D()
    max_1_2 = L.MaxPooling1D()
    max_1_3 = L.MaxPooling1D()

    conv4_1_known = max_1_1(conv4_1(known_embed))
    conv8_1_known = max_1_2(conv8_1(known_embed))
    conv16_1_known = max_1_3(conv16_1(known_embed))

    conv4_1_unknown = max_1_1(conv4_1(unknown_embed))
    conv8_1_unknown = max_1_2(conv8_1(unknown_embed))
    conv16_1_unknown = max_1_3(conv16_1(unknown_embed))

    # Convolution 2

    conv8_2 = L.Convolution1D(filters=500, kernel_size=8, strides=1,
                              activation='relu', name='convolutional_8_2')

    conv4_2 = L.Convolution1D(filters=500, kernel_size=4, strides=1,
                              activation='relu', name='convolutional_4_2')

    conv16_2 = L.Convolution1D(filters=500, kernel_size=16, strides=1,
                               activation='relu', name='convolutional_16_2')

    max_2_1 = L.MaxPooling1D()
    max_2_2 = L.MaxPooling1D()
    max_2_3 = L.MaxPooling1D()

    conv4_2_known = max_2_1(conv4_2(conv4_1_known))
    conv8_2_known = max_2_2(conv8_2(conv8_1_known))
    conv16_2_known = max_2_3(conv16_2(conv16_1_known))

    conv4_2_unknown = max_2_1(conv4_2(conv4_1_unknown))
    conv8_2_unknown = max_2_2(conv8_2(conv8_1_unknown))
    conv16_2_unknown = max_2_3(conv16_2(conv16_1_unknown))

    # Convolution 3
    conv8_3 = L.Convolution1D(filters=500, kernel_size=8, strides=1,
                              activation='relu', name='convolutional_8_3')

    conv4_3 = L.Convolution1D(filters=500, kernel_size=4, strides=1,
                              activation='relu', name='convolutional_4_3')

    conv16_3 = L.Convolution1D(filters=500, kernel_size=16, strides=1,
                               activation='relu', name='convolutional_16_3')

    # Glob Max Pool
    repr_known1 = L.GlobalMaxPooling1D(name='known_repr_4')(
        conv4_3(conv4_2_known))
    repr_unknown1 = L.GlobalMaxPooling1D(name='unknown_repr_4')(
        conv4_3(conv4_2_unknown))

    repr_known2 = L.GlobalMaxPooling1D(name='known_repr_8')(
        conv8_3(conv8_2_known))
    repr_unknown2 = L.GlobalMaxPooling1D(name='unknown_repr_8')(
        conv8_3(conv8_2_unknown))

    repr_known3 = L.GlobalMaxPooling1D(name='known_repr_8_2')(
        conv16_3(conv16_2_known))
    repr_unknown3 = L.GlobalMaxPooling1D(name='unknown_repr_8_2')(
        conv16_3(conv16_2_unknown))

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
    dense2 = L.Dense(500, activation='relu')(dense1)
    dense3 = L.Dense(500, activation='relu')(dense2)
    dense4 = L.Dense(500, activation='relu')(dense3)

    pruned = L.Dropout(0.3)(dense4)

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
