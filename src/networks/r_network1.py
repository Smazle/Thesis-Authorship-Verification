#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import keras.backend as K
import jsonpickle
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Concatenate, Input, Dropout, GRU, Reshape,\
    Lambda, Flatten, Activation, merge
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
        batch_size=1,
        vocabulary_frequency_cutoff=1 / 100000,
        validation_split=0.95
    )

if args.reader is not None:
    with open('reader.p', mode='w') as reader_out:
        reader_out.write(jsonpickle.encode(reader))

inshape = (reader.max_len, )

known_in = Input(shape=inshape, name='Known_Input')
unknown_in = Input(shape=inshape, name='Unknown_Input')

embedding = Embedding(len(reader.vocabulary_above_cutoff) + 2, 5)

known_emb = embedding(known_in)
unknown_emb = embedding(unknown_in)

gru = GRU(100, activation='relu')

gru_known = gru(known_emb)
gru_unknown = gru(unknown_emb)

known_out = Activation('softmax')(gru_known)
unknown_out = Activation('softmax')(gru_unknown)

cos_distance = merge([known_out, unknown_out], mode='cos', dot_axes=1)
cos_distance = Reshape((1,))(cos_distance)
cos_similarity = Lambda(lambda x: 1 - x)(cos_distance)

output = Dense(2, activation='softmax')(cos_similarity)

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
    epochs=1,
    validation_data=reader.generate_validation(),
    validation_steps=val_steps_n
)

model.save('final_model.hdf5')
