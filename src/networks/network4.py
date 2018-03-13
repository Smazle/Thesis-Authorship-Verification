# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np
import argparse
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Concatenate, Input, Dropout, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from ..preprocessing import MacomReader
# from ..util import utilities as util

np.random.seed(7)

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to data file.'
)
parser.add_argument(
    '--history',
    type=str,
    help='Path to file to write history to.',
    default=None
)
parser.add_argument(
    '--graph',
    type=str,
    help='Path to file to visualize network in.'
)
args = parser.parse_args()


reader = MacomReader(
    args.datafile,
    batch_size=8,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000,
    char=False
)

TopWords = 500
EmbeddingsLength = 32

with reader as generator:
    inshape = (generator.max_len, )

    first = Input(shape=inshape)
    second = Input(shape=inshape)

    concat = Concatenate()([first, second])

    emb = Embedding(len(generator.vocabulary_above_cutoff) + 2,
                    EmbeddingsLength,
                    input_length=generator.max_len * 2)(concat)
    d1 = Dropout(0.2)(emb)
    lst = GRU(100)(d1)
    d2 = Dropout(0.2)(lst)
    output = Dense(2, activation='softmax')(d2)

    model = Model(inputs=[first, second], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if args.graph is not None:
        plot_model(model, to_file=args.graph, show_shapes=True)

    steps_n = len(generator.training_problems) / generator.batch_size
    val_steps_n = len(generator.validation_problems) / generator.batch_size

    callbacks = [
        ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    model.fit_generator(
        generator=generator.generate_training(),
        steps_per_epoch=steps_n,
        epochs=100,
        validation_data=generator.generate_validation(),
        validation_steps=val_steps_n,
        callbacks=callbacks)
