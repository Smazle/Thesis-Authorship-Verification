#!/usr/bin/python3
# -*- coding: utf-8 -*-

from keras.layers import Dense, Convolution1D, GlobalMaxPooling1D, Input,\
    Concatenate, Embedding
from keras.models import Model
from ..preprocessing import MacomReader
from keras.utils import plot_model
import argparse
import resource
from ..util import CSVWriter
from keras.callbacks import ModelCheckpoint


gb4 = 4000000000  # 4 GB in bytes.
gb6 = 6000000000  # 6 GB in bytes.

# Limit memory usage of the script so we don't crash a computer.
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (gb6, hard))

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument('datafile', type=str, help='Path to data file.')
args = parser.parse_args()

reader = MacomReader(
    args.datafile,
    batch_size=2,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000
)

with reader as generator:
    inshape = (generator.max_len, )

    known_in = Input(shape=inshape)
    unknown_in = Input(shape=inshape)

    embedding = Embedding(len(generator.vocabulary_above_cutoff) + 2, 5,
                          input_length=generator.max_len)

    conv = Convolution1D(filters=1000, kernel_size=10, strides=1,
                         activation='relu')

    repr_known = GlobalMaxPooling1D()(conv(embedding(known_in)))
    repr_unknown = GlobalMaxPooling1D()(conv(embedding(unknown_in)))

    full_input = Concatenate()([repr_known, repr_unknown])

    dense1 = Dense(500, activation='relu')(full_input)
    dense2 = Dense(500, activation='relu')(dense1)
    output = Dense(2, activation='softmax')(dense2)

    model = Model(inputs=[known_in, unknown_in], outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='siamese.png')

    steps_n = len(generator.training_problems) / generator.batch_size
    val_steps_n = len(generator.validation_problems) / generator.batch_size

    callbacks = [
        ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True
        ),
        CSVWriter(generator.generate_validation(), val_steps_n, 'history.csv')
    ]

    model.fit_generator(
        generator=generator.generate_training(),
        steps_per_epoch=steps_n,
        epochs=100,
        validation_data=generator.generate_validation(),
        validation_steps=val_steps_n,
        callbacks=callbacks
    )
