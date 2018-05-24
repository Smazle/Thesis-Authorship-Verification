#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..preprocessing.channels import ChannelType
from keras.models import Model
import keras.backend as K
import keras.layers as L
import keras.optimizers as O


def model(reader):
    assert reader.channeltypes == [ChannelType.SENTENCE]

    sentence_len = reader.channels[0].sentence_len

    known_in = L.Input(shape=(None, sentence_len), dtype='int32')
    unknown_in = L.Input(shape=(None, sentence_len), dtype='int32')

    # TODO: Path should be command line argument or something.
    weights = gew.GetEmbeddingWeights(
        '/home/smazle/Git/MastersThesis/data/pre-trained/wiki.da.vec', reader)

    embedding = L.Embedding(output_dim=weights.shape[1],
                            input_dim=weights.shape[0], trainable=False,
                            weights=[weights])

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    known_sentences_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sentence_len, output_shape=(None, 300))(known_emb)
    unknown_sentences_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sentence_len, output_shape=(None, 300))(unknown_emb)

    feature_extractor = L.Bidirectional(L.LSTM(50, return_sequences=True))
    learn_from_features = L.Bidirectional(L.LSTM(50, return_sequences=True))

    features_known = L.GlobalAvgPool1D()(learn_from_features(feature_extractor(known_sentences_repr)))
    features_unknown = L.GlobalAvgPool1D()(learn_from_features(feature_extractor(unknown_sentences_repr)))

    abs_diff = L.merge(
        inputs=[features_known, features_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    dense1 = L.Dense(100)

    pruned = L.Dropout(0.3)(dense1(abs_diff))

    output = L.Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    optimizer = O.Adam(lr=0.0005)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
