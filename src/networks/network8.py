#!/usr/bin/python3
# -*- coding: utf-8 -*-

from src.preprocessing.channels import ChannelType
import keras.layers as L
import keras.backend as K
from keras.models import Model
from src.util import generate_emb_weight as gew


def model(reader):
    assert reader.channeltypes == [
        ChannelType.CHAR, ChannelType.WORD, ChannelType.SENTENCE
    ]

    sent_len = reader.channels[-1].sentence_len

    text_1_char = L.Input(shape=(None, ), dtype='int32')
    text_1_word = L.Input(shape=(None, ), dtype='int32')
    text_1_sent = L.Input(shape=(None, sent_len), dtype='int32')

    text_2_char = L.Input(shape=(None, ), dtype='int32')
    text_2_word = L.Input(shape=(None, ), dtype='int32')
    text_2_sent = L.Input(shape=(None, sent_len), dtype='int32')

    word_mapping = reader.channels[1].vocabulary_map

    word_weights = gew.generate_embedding_weights(word_mapping)

    word_number = word_weights.shape[0]
    word_embedding_size = word_weights.shape[1]

    char_embedding = L.Embedding(
        len(reader.channels[0].vocabulary_above_cutoff) + 2, 5)

    word_embedding = L.Embedding(
        output_dim=word_embedding_size,
        input_dim=word_number,
        trainable=False,
        weights=[word_weights])

    sent_embedding = L.Embedding(
        output_dim=word_weights.shape[1],
        input_dim=word_weights.shape[0],
        trainable=False,
        weights=[word_weights])

    # Embed all inputs.
    text_1_char_emb = char_embedding(text_1_char)
    text_1_word_emb = word_embedding(text_1_word)
    text_1_sent_emb = sent_embedding(text_1_sent)
    text_2_char_emb = char_embedding(text_2_char)
    text_2_word_emb = word_embedding(text_2_word)
    text_2_sent_emb = sent_embedding(text_2_sent)

    # Create sentence representations
    text_1_sent_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sent_len,
        output_shape=(None, 300))(text_1_sent_emb)
    text_2_sent_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sent_len,
        output_shape=(None, 300))(text_2_sent_emb)

    # Define convolutions.
    char_conv8 = L.Convolution1D(
        filters=200, kernel_size=8, strides=1, activation='relu')
    char_conv4 = L.Convolution1D(
        filters=200, kernel_size=4, strides=1, activation='relu')
    word_conv8 = L.Convolution1D(
        filters=100, kernel_size=8, strides=1, activation='relu')
    sent_conv4 = L.Convolution1D(
        filters=200, kernel_size=4, strides=1, activation='relu')

    # Convolve all inputs and pool them to a representation of the texts.
    text_1_char_repr_8 = L.GlobalMaxPooling1D()(char_conv8(text_1_char_emb))
    text_1_char_repr_4 = L.GlobalMaxPooling1D()(char_conv4(text_1_char_emb))
    text_1_word_repr_8 = L.GlobalMaxPooling1D()(word_conv8(text_1_word_emb))
    text_1_sent_repr_4 = L.GlobalMaxPooling1D()(sent_conv4(text_1_sent_repr))

    text_2_char_repr_8 = L.GlobalMaxPooling1D()(char_conv8(text_2_char_emb))
    text_2_char_repr_4 = L.GlobalMaxPooling1D()(char_conv4(text_2_char_emb))
    text_2_word_repr_8 = L.GlobalMaxPooling1D()(word_conv8(text_2_word_emb))
    text_2_sent_repr_4 = L.GlobalMaxPooling1D()(sent_conv4(text_2_sent_repr))

    # Representation of texts are concatenation of all representations.
    text_1_repr = L.Concatenate()([
        text_1_char_repr_8, text_1_char_repr_4, text_1_word_repr_8,
        text_1_sent_repr_4
    ])
    text_2_repr = L.Concatenate()([
        text_2_char_repr_8, text_2_char_repr_4, text_2_word_repr_8,
        text_2_sent_repr_4
    ])

    # Find the difference between the texts and learn on it.
    abs_diff = L.merge(
        inputs=[text_1_repr, text_2_repr],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference')

    dense1 = L.Dense(500, activation='relu')(abs_diff)
    dense2 = L.Dense(500, activation='relu')(dense1)
    dense3 = L.Dense(500, activation='relu')(dense2)
    dense4 = L.Dense(500, activation='relu')(dense3)

    pruned = L.Dropout(0.3)(dense4)

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(
        inputs=[
            text_1_char, text_1_word, text_1_sent, text_2_char, text_2_word,
            text_2_sent
        ],
        outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
