#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import jsonpickle
import keras
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Concatenate, Input, Dropout, GRU, Reshape,\
    Lambda, Flatten, Activation, merge, Convolution1D, MaxPooling1D, LSTM, CuDNNGRU
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.layers.pooling import AveragePooling1D
from ..preprocessing import MacomReader
from sklearn.metrics.pairwise import cosine_similarity
from ..util import CSVWriter

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple RNN for authorship verification'
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
parser.add_argument(
    '--reader',
    type=str,
    help='Use this pickled reader and not a new reader.'
)
parser.add_argument(
    '--weights',
    type=str,
    help='Use the weights given as start weights instead of randomly' +
         ' initializing.'
)
args = parser.parse_args()

# Either load reader from file or create a new one.
if args.reader is not None:
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())
else:
    reader = MacomReader(
        args.datafile,
        batch_size=8,
        vocabulary_frequency_cutoff=1 / 100000,
        validation_split=0.95,
        pad=False,
        batch_normalization="pad"
    )

    with open('reader.p', mode='w') as reader_out:
        reader_out.write(jsonpickle.encode(reader))

inshape = (None, )

known_in = Input(shape=inshape, dtype='int32')
unknown_in = Input(shape=inshape, dtype='int32')

embedding = Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)

known_emb = embedding(known_in)
unknown_emb = embedding(unknown_in)

conv8 = Convolution1D(filters=500, kernel_size=8, strides=1,
                      activation='relu', padding='same')
pool = MaxPooling1D(pool_size=8)
gru = CuDNNGRU(100)

features_known = gru(pool(conv8(known_emb)))
features_unknown = gru(pool(conv8(unknown_emb)))

abs_diff = merge(
    inputs=[features_known, features_unknown],
    mode=lambda x: abs(x[0] - x[1]),
    output_shape=lambda x: x[0],
    name='absolute_difference'
)

dense1 = Dense(500, activation='relu')(abs_diff)
dense2 = Dense(500, activation='relu')(dense1)

pruned = Dropout(0.3)(dense1)

output = Dense(2, activation='softmax', name='output')(pruned)

model = Model(inputs=[known_in, unknown_in], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_n = len(reader.training_problems) / reader.batch_size
val_steps_n = len(reader.validation_problems) / reader.batch_size

# Setup callbacks.
callbacks = [
    ModelCheckpoint(
        'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True
    )
]

if args.history is not None:
    callbacks.append(
        CSVWriter(
            reader.generate_validation(), val_steps_n,
            reader.generate_training(), steps_n, args.history,
            args.weights is not None
        )
    )

# If we are asked to visualize model, do so.
if args.graph is not None:
    plot_model(model, to_file=args.graph, show_shapes=True)

# If we are given weights, load them.
if args.weights is not None:
    model.load_weights(args.weights)

model.fit_generator(
    generator=reader.generate_training(),
    steps_per_epoch=steps_n,
    epochs=100,
    validation_data=reader.generate_validation(),
    validation_steps=val_steps_n,
    callbacks=callbacks
)

model.save('final_model.hdf5')
