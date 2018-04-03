#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras import backend as K
import argparse
import jsonpickle
from ..preprocessing import LineReader
from ..util import utilities as util


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
parser.add_argument(
    '--convolution-size',
    type=int,
    help='Size of the convolutional layer we are looking at.',
    default=8
)
parser.add_argument(
    '--layer-name',
    type=str,
    help='Which layer to get the output from.',
    default='convolutional_8'
)
parser.add_argument(
    '--filter',
    type=int,
    help='Which filter number to look at.',
    default=0
)
parser.add_argument(
    '--outfile',
    type=str,
    help='Write the output as a CSV file in this location.',
    default=None
)
args = parser.parse_args()

model = load_model(args.model)

get_output = K.function([
    model.get_layer('known_input').input,
    model.get_layer('unknown_input').input,
    K.learning_phase()], [
    model.get_layer(args.layer_name).get_output_at(0)]
)

with open(args.reader, 'r') as macomreader_in:
    reader = jsonpickle.decode(macomreader_in.read())
    reader.batch_size = 1

output = []
with LineReader(args.datafile, encoding='utf-8') as linereader:
    opposition = reader.read_encoded_line(linereader, 1)
    for i in range(1, len(linereader.line_offsets)):
        text1 = reader.read_encoded_line(linereader, i)

        text1 = np.expand_dims(text1, axis=0)
        text2 = np.expand_dims(opposition, axis=0)

        layer_output = get_output([text1, text2, 0])[0]

        _filter = layer_output[0, :, args.filter]
        max_ind = np.argmax(_filter)
        max_val = np.max(_filter)

        author, date, text = linereader.readline(i).split(';')
        text = util.clean(text)

        max_text = repr(text[max_ind:max_ind + args.convolution_size])
        print(max_text, max_val)

        output.append((i, max_text, max_val))

output.sort(key=lambda x: x[2], reverse=True)

if args.outfile is not None:
    with open(args.outfile, 'w') as f:
        f.write('text,max-string,max-val\r\n')
        for i, string, val in output:
            f.write('{},{},{}\r\n'.format(i, string, val))
