#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras import backend as K
import argparse
import jsonpickle
from ..preprocessing import LineReader
from ..util import utilities as util
import sys


# Global variables initialized in "initialize_globals".

# Keras function that returns output of convolutional_8 layer.
GET_OUTPUT_8_1 = None

# Keras function that returns output of convolutional_8_2 layer.
GET_OUTPUT_8_2 = None

# Keras function that returns output of convolutional_4 layer.
GET_OUTPUT_4 = None


def main():
    args = parse_args()
    model = load_model(args.model)
    initialize_globals(model)

    with open(args.reader, 'r') as macomreader_in:
        reader = jsonpickle.decode(macomreader_in.read())
        reader.batch_size = 1

    output = []
    with LineReader(args.datafile, encoding='utf-8') as linereader:
        opposition = reader.read_encoded_line(linereader, 1)
        lines = len(linereader.line_offsets)
        print('text,filter,activation_string,activation_value', end='\r\n')
        for i in range(1, lines):
            print('Handling text {} of {}'.format(i, lines), file=sys.stderr)

            text1 = reader.read_encoded_line(linereader, i)

            text1 = list(map(lambda x: np.expand_dims(x, axis=0), text1))
            text2 = list(map(lambda x: np.expand_dims(x, axis=0), opposition))

            author, date, text = linereader.readline(i).split(';')
            text = util.clean(text)

            handle_4(text1 + text2 + [0], text, i)
            handle_8(text1 + text2 + [0], text, i)


def handle_4(input_texts, text, text_number):
    layer_output = GET_OUTPUT_4(input_texts)[0]

    # Go through the 500 filters in convolutional_8.
    for i in range(500):
        index = i + 700  # Add the 700 8 filters.
        _filter = layer_output[0, :, i]
        max_ind = np.argmax(_filter)
        max_val = np.max(_filter)
        max_text = repr(text[max_ind:max_ind + 4])

        print('{},{},{},{},{}'.format(text_number, index, max_ind, max_text,
                                      max_val), end='\r\n')


def handle_8(input_texts, text, text_number):
    layer_output_1 = GET_OUTPUT_8_1(input_texts)[0]
    layer_output_2 = GET_OUTPUT_8_2(input_texts)[0]

    # Go through the 500 filters in convolutional_8.
    for i in range(500):
        _filter = layer_output_1[0, :, i]
        max_ind = np.argmax(_filter)
        max_val = np.max(_filter)
        max_text = repr(text[max_ind:max_ind + 8])

        print('{},{},{},{},{}'.format(text_number, i, max_ind, max_text,
                                      max_val), end='\r\n')

    # Go through the 200 filters in convolutional_8_2.
    for i in range(200):
        index = i + 500  # Add the 500 8 filters.
        _filter = layer_output_2[0, :, i]
        max_ind = np.argmax(_filter)
        max_val = np.max(_filter)
        max_text = repr(text[max_ind:max_ind + 8])

        print('{},{},{},{},{}'.format(text_number, index, max_ind, max_text,
                                      max_val), end='\r\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Output what the filters react to in each text.')
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
    parser.add_argument(
        '--convolution-size',
        type=int,
        help='Size of the convolutional layer we are looking at.',
        default=8
    )
    parser.add_argument(
        '--outfile',
        type=str,
        help='Write the output as a CSV file in this location.',
        default=None
    )
    args = parser.parse_args()

    return args


def initialize_globals(model):
    global GET_OUTPUT_8_1
    GET_OUTPUT_8_1 = K.function([
        model.get_layer('known_input').input,
        model.get_layer('unknown_input').input,
        K.learning_phase()
    ], [model.get_layer('convolutional_8').get_output_at(0)])

    global GET_OUTPUT_8_2
    GET_OUTPUT_8_2 = K.function([
        model.get_layer('known_input').input,
        model.get_layer('unknown_input').input,
        K.learning_phase()
    ], [model.get_layer('convolutional_8_2').get_output_at(0)])

    global GET_OUTPUT_4
    GET_OUTPUT_4 = K.function([
        model.get_layer('known_input').input,
        model.get_layer('unknown_input').input,
        K.learning_phase()
    ], [model.get_layer('convolutional_4').get_output_at(0)])


if __name__ == '__main__':
    main()
