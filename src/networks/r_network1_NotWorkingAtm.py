# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np
import argparse
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Concatenate, Input, Dropout, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.layers.pooling import AveragePoolingID
from ..preprocessing import MacomReader
# from ..util import utilities as util

np.random.seed(7)

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
# if args.reader is not None:
#    with open(args.reader, mode='r') as reader_in:
#        reader = jsonpickle.decode(reader_in.read())
# else:
reader = MacomReader(
    args.datafile,
    batch_size=2,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000,
    validation_split=0.95
)

# if args.reader is not None:
#    with open('reader.p', mode='w') as reader_out:
#        reader_out.write(jsonpickle.encode(reader))

inshape = (reader.max_len, )

known_in = Input(shape=inshape, name='Known Input')
unknown_in = Input(shape=inshape, name='Unknown Input')

embedding = Embedding(len(reader.vocabulary_frequency_cutoff) + 2, 5,
                      input_length=inshape[0])

known_emb = embedding(known_in)
unknown_in = embedding(unknown_in)

gru = GRU(100, activation='relu')

gru_known = gru(known_emb)
gru_unknown = gru(unknown_in)

avg_known = AveragePoolingID(pool_size=4)(gru_known)
avg_unknown = AveragePoolingID(pool_size=4)(gru_unknown)

output = Dense('--NumClasses--', activation='softmax')
