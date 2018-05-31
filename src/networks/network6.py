#!/usr/bin/python3
# -*- coding: utf-8 -*-

from src.preprocessing.channels import ChannelType
import keras.layers as L
from keras.models import Model
from src.util import generate_emb_weight as gew


def model(reader):
    assert reader.channeltypes == [ChannelType.CHAR, ChannelType.WORD]

    text_1_char = L.Input(shape=(None, ), dtype='int32')
    text_1_word = L.Input(shape=(None, ), dtype='int32')
    text_2_char = L.Input(shape=(None, ), dtype='int32')
    text_2_word = L.Input(shape=(None, ), dtype='int32')

    word_mapping = reader.channels[1].vocabulary_map

    word_weights = gew.generate_embedding_weights(
        '.\\data\\pre-trained\\wiki.da.vec', word_mapping)

    word_number = word_weights.shape[0]
    word_embedding_size = word_weights.shape[1]

    char_embedding = L.Embedding(
        len(reader.channels[0].vocabulary_above_cutoff) + 2, 5
    )

    word_embedding = L.Embedding(
        output_dim=word_embedding_size,
        input_dim=word_number,
        trainable=False,
        weights=[word_weights]
    )

    # Embed all inputs.
    text_1_char_emb = char_embedding(text_1_char)
    text_1_word_emb = word_embedding(text_1_word)
    text_2_char_emb = char_embedding(text_2_char)
    text_2_word_emb = word_embedding(text_2_word)

    # Define convolutions.
    char_conv8 = L.Convolution1D(
        filters=200, kernel_size=8, strides=1, activation='relu')
    char_conv4 = L.Convolution1D(
        filters=200, kernel_size=4, strides=1, activation='relu')
    word_conv8 = L.Convolution1D(
        filters=100, kernel_size=8, strides=1, activation='relu')

    # Convolve all inputs and pool them to a representation of the texts.
    text_1_char_repr_8 = L.GlobalMaxPooling1D()(char_conv8(text_1_char_emb))
    text_1_char_repr_4 = L.GlobalMaxPooling1D()(char_conv4(text_1_char_emb))
    text_1_word_repr_8 = L.GlobalMaxPooling1D()(word_conv8(text_1_word_emb))

    text_2_char_repr_8 = L.GlobalMaxPooling1D()(char_conv8(text_2_char_emb))
    text_2_char_repr_4 = L.GlobalMaxPooling1D()(char_conv4(text_2_char_emb))
    text_2_word_repr_8 = L.GlobalMaxPooling1D()(word_conv8(text_2_word_emb))

    # Representation of texts are concatenation of all representations.
    text_1_repr = L.Concatenate()(
        [text_1_char_repr_8, text_1_char_repr_4, text_1_word_repr_8])
    text_2_repr = L.Concatenate()(
        [text_2_char_repr_8, text_2_char_repr_4, text_2_word_repr_8])

    # Find the difference between the texts and learn on it.
    abs_diff = L.merge(
        inputs=[text_1_repr, text_2_repr],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    dense1 = L.Dense(500, activation='relu')(abs_diff)
    dense2 = L.Dense(500, activation='relu')(dense1)

    pruned = L.Dropout(0.3)(dense2)

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(
        inputs=[text_1_char, text_1_word, text_2_char, text_2_word],
        outputs=output
    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
