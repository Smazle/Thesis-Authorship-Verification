#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import keras.backend as K
import jsonpickle
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Concatenate, Input, Dropout, GRU, Reshape,\
    Lambda, Flatten, Activation
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



def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B

    A_mag = l2_norm(A, axis=1)
    B_mag = l2_norm(B, axis=1)
    num = K.batch_dot(K.permute_dimensions(B, (0,2,1)), A)
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den

    print(num.get_shape())
    print(den.get_shape())
    print(dist_mat.get_shape())

    return dist_mat

def shape(input_shape):
    shape1, shape2 = input_shape
    return (shape1[0], 1)


# Either load reader from file or create a new one.
if args.reader is not None:
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())
else:
    reader = MacomReader(
        args.datafile,
        batch_size=1,
        encoding='numbers',
        vocabulary_frequency_cutoff=1 / 100000,
        validation_split=0.95
    )

if args.reader is not None:
    with open('reader.p', mode='w') as reader_out:
        reader_out.write(jsonpickle.encode(reader))

inshape = (reader.max_len, )

known_in = Input(shape=inshape, name='Known_Input')
unknown_in = Input(shape=inshape, name='Unknown_Input')

embedding = Embedding(len(reader.vocabulary_above_cutoff) + 2, 5,
                      input_length=inshape[0])

known_emb = embedding(known_in)
unknown_emb = embedding(unknown_in)

gru = GRU(100, activation='relu')

gru_known = gru(known_emb)
gru_unknown = gru(unknown_emb)

gru_known = Reshape((100, 1))(gru_known)
gru_unknown = Reshape((100, 1))(gru_unknown)

avg_known = AveragePooling1D()(gru_known)
avg_unknown = AveragePooling1D()(gru_unknown)

#avg_known = Flatten()(avg_known)
#avg_unknown = Flatten()(avg_unknown)

known_out = Activation("softmax")(avg_known)
unknown_out = Activation("softmax")(avg_unknown)

cosine_sim = Lambda(pairwise_cosine_sim, output_shape=shape)([known_out, unknown_out])

#output = Flatten()(cosine_sim)

#output = Activation("softmax")(cosine_sim)

model = Model(inputs=[known_in, unknown_in], outputs=cosine_sim)



model.compile(optimizer='adam',
              loss='mse',
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
