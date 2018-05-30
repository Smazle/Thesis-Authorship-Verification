#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..preprocessing.channels import ChannelType
from keras.models import Model
import keras.backend as K
import keras.layers as L
from ..util import generate_emb_weight as gew


def model(reader):
    assert reader.channeltypes == [ChannelType.SENTENCE]

    sent_len = reader.channels[0].sentence_len
    weights = gew.GetEmbeddingWeights(
        '/home/smazle/Git/MastersThesis/data/pre-trained/wiki.da.vec', reader)

    embedding = L.Embedding(output_dim=weights.shape[1],
                            input_dim=weights.shape[0], trainable=False,
                            weights=[weights])

    known_in = L.Input(shape=(None, sent_len), dtype='int32')
    unknown_in = L.Input(shape=(None, sent_len), dtype='int32')

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    known_sentences_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sent_len,
        output_shape=(None, 300))(known_emb)
    unknown_sentences_repr = L.Lambda(
        lambda x: K.sum(x, axis=2) / sent_len,
        output_shape=(None, 300))(unknown_emb)

    feature_extractor = L.Bidirectional(L.GRU(10))

    features_known = feature_extractor(known_sentences_repr)
    features_unknown = feature_extractor(unknown_sentences_repr)

    abs_diff = L.merge(
        inputs=[features_known, features_unknown],
        mode=lambda x: abs(x[0] - x[1]),
        output_shape=lambda x: x[0],
        name='absolute_difference'
    )

    output = L.Dense(2, activation='softmax', name='output')(abs_diff)

    model = Model(inputs=[known_in, unknown_in], outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
