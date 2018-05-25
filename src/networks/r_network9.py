#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..preprocessing.channels import ChannelType
from keras.models import Model
import keras.backend as K
import keras.layers as L
import keras.optimizers as O
from ..util import generate_emb_weight as gew


def model(reader):
    assert reader.channeltypes == [ChannelType.SENTENCE]

    word_mapping = reader.channels[0].vocabulary_map

    # TODO: Path should be command line argument or something.
    weights = gew.generate_embedding_weights(
        '/home/fluttershy/datalogi/masters_project/MastersThesis/data/pre-trained/wiki.da.vec', word_mapping)

    sentence_len = reader.channels[0].sentence_len
    word_number = weights.shape[0]
    word_embedding_size = weights.shape[1]

    known_in = L.Input(shape=(None, sentence_len), dtype='int32')
    unknown_in = L.Input(shape=(None, sentence_len), dtype='int32')

    embedding = L.Embedding(
        output_dim=word_embedding_size,
        input_dim=word_number,
        trainable=False,
        weights=[weights]
    )

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    average_sentences = L.Lambda(
        lambda x: K.sum(x, axis=2) / sentence_len,
        output_shape=(None, word_embedding_size)
    )

    known_sentences_repr = average_sentences(known_emb)
    unknown_sentences_repr = average_sentences(unknown_emb)

    feature_extractor1 = L.Bidirectional(L.LSTM(50, return_sequences=True))
    feature_extractor2 = L.Bidirectional(L.LSTM(50, return_sequences=True))

    pool = L.GlobalAvgPool1D()

    features_known = pool(
        feature_extractor2(feature_extractor1(known_sentences_repr)))
    features_unknown = pool(
        feature_extractor2(feature_extractor1(unknown_sentences_repr)))

    abs_diff = L.merge(
        inputs=[features_known, features_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    # Dense network.
    dense1 = L.Dense(100)(abs_diff)
    pruned = L.Dropout(0.3)(dense1)
    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    optimizer = O.Adam(lr=0.0005)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
