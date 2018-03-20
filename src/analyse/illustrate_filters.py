#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras import backend as K
import argparse
import jsonpickle
from ..preprocessing import LineReader


# Parse arguments.
parser = argparse.ArgumentParser(
    description='Output what the filters react to in each text.'
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to data file.'
)
parser.add_argument(
    'reader',
    type=str,
    help='Path to reader that can read and encode the lines.'
)
parser.add_argument(
    'model',
    type=str,
    help='Path to model that can predict.'
)
args = parser.parse_args()


model = load_model(args.model)

get_output = K.function([
    model.get_layer('known_input').input,
    model.get_layer('unknown_input').input,
    K.learning_phase()], [
    model.get_layer('convolutional_8').get_output_at(0)]
)

with open(args.reader, 'r') as macomreader_in:
    reader = jsonpickle.decode(macomreader_in.read())
    reader.batch_size = 1

opposition = 2
conv_size = 8

with LineReader(args.datafile, encoding='utf-8') as linereader:
    for i in range(1, len(linereader.line_offsets)):
        text1 = reader.read_encoded_line(linereader, i)
        text2 = reader.read_encoded_line(linereader, opposition)

        text1 = np.expand_dims(text1, axis=0)
        text2 = np.expand_dims(text2, axis=0)

        layer_output = get_output([text1, text2, 0])[0]

        first_filter = layer_output[0, :, 0]
        max_ind = np.argmax(first_filter)

        text = linereader.readline(i)\
            .split(';')[1]\
            .replace('$NL$', '\n')\
            .replace('$SC$', ';')

        print(repr(text[max_ind:max_ind + 8]))
