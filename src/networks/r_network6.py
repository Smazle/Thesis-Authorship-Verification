
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
import keras.layers as L
import keras.backend as K


def model(reader):
    known_in = L.Input(shape=(None, 25), dtype='int32')
    unknown_in = L.Input(shape=(None, 25), dtype='int32')

    embedding = L.Embedding(len(reader.channels[0].vocabulary_above_cutoff) + 2,
            150)

    known_emb = embedding(known_in)
    unknown_emb = embedding(unknown_in)

    known_sentences_repr = L.Lambda(lambda x: K.sum(x, axis=2),
            output_shape=(None, 150))(known_emb)
    unknown_sentences_repr = L.Lambda(lambda x: K.sum(x, axis=2),
            output_shape=(None, 150))(unknown_emb)

    feature_extractor = L.Bidirectional(L.LSTM(50))

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



















    # known_emb = embedding(known_in)
    # unknown_emb = embedding(unknown_in)

    # feature_extractor = L.Bidirectional(L.LSTM(500))

    # features_known = feature_extractor(known_emb)
    # features_unknown = feature_extractor(unknown_emb)

    # abs_diff = L.merge(
        # inputs=[features_known, features_unknown],
        # mode=lambda x: abs(x[0] - x[1]),
        # output_shape=lambda x: x[0],
        # name='absolute_difference'
    # )

    # output = L.Dense(2, activation='softmax', name='output')(abs_diff)

    # model = Model(inputs=[known_in, unknown_in], outputs=output)

    # model.compile(optimizer='adam',
                # loss='categorical_crossentropy',
                # metrics=['accuracy'])

    # return model
