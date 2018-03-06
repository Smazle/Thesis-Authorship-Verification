#!/usr/bin/python3
# -*- coding: utf-8 -*-

from keras.layers import Dense, Convolution1D, GlobalMaxPooling1D, Input,\
    Concatenate, Embedding, Dropout
from keras.models import Model
from ..preprocessing import MacomReader
import argparse
from ..util import CSVWriter
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


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
    batch_size=2,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000
)

with reader as generator:
    inshape = (generator.max_len, )

    known_in = Input(shape=inshape, name='known_input')
    unknown_in = Input(shape=inshape, name='unknown_input')

    embedding = Embedding(len(generator.vocabulary_above_cutoff) + 2, 5,
                          input_length=generator.max_len)

    known_embed = embedding(known_in)
    unknown_embed = embedding(unknown_in)

    conv8 = Convolution1D(filters=500, kernel_size=8, strides=1,
                          activation='relu', name='convolutional_8')

    conv4 = Convolution1D(filters=500, kernel_size=4, strides=1,
                          activation='relu', name='convolutional_4')

    repr_known1 = GlobalMaxPooling1D(name='known_repr_8')(
        conv8(known_embed))
    repr_unknown1 = GlobalMaxPooling1D(name='unknown_repr_8')(
        conv8(unknown_embed))

    repr_known2 = GlobalMaxPooling1D(name='known_repr_4')(
        conv4(known_embed))
    repr_unknown2 = GlobalMaxPooling1D(name='unknown_repr_4')(
        conv4(unknown_embed))

    full_input = Concatenate()(
        [repr_known1, repr_known2, repr_unknown1, repr_unknown2])

    dense1 = Dense(500, activation='relu')(full_input)
    dense2 = Dense(500, activation='relu')(dense1)

    pruned = Dropout(0.3)(dense2)

    output = Dense(2, activation='softmax', name='output')(pruned)

    model = Model(inputs=[known_in, unknown_in], outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    steps_n = len(generator.training_problems) / generator.batch_size
    val_steps_n = len(generator.validation_problems) / generator.batch_size

    # Setup callbacks.
    callbacks = [
        ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    if args.history is not None:
        callbacks.append(
            CSVWriter(
                generator.generate_validation(), val_steps_n, args.history
            )
        )

    # If we are asked to visualize model, do so.
    if args.graph is not None:
        plot_model(model, to_file=args.graph)

    model.fit_generator(
        generator=generator.generate_training(),
        steps_per_epoch=steps_n,
        epochs=100,
        validation_data=generator.generate_validation(),
        validation_steps=val_steps_n,
        callbacks=callbacks
    )
