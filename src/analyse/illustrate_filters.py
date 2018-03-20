#!/usr/bin/env python3

import numpy as np
from keras.models import load_model
from keras import backend as K
import jsonpickle
import sys
from ..preprocessing import LineReader


model = load_model(sys.argv[1])

get_output = K.function([
    model.get_layer('known_input').input,
    model.get_layer('unknown_input').input,
    K.learning_phase()], [
    model.get_layer('convolutional_8').get_output_at(0)]
)

with open(sys.argv[2], 'r') as macomreader_in:
    reader = jsonpickle.decode(macomreader_in.read())
    reader.batch_size = 1

text_line = 32
opposition = 2
conv_size = 8

with LineReader(sys.argv[3], encoding='utf-8') as linereader:
    text1 = reader.read_encoded_line(linereader, text_line)
    text2 = reader.read_encoded_line(linereader, opposition)

    text1 = np.expand_dims(text1, axis=0)
    text2 = np.expand_dims(text2, axis=0)

    layer_output = get_output([text1, text2, 0])[0]

    # Extract first filter.
    first_filter = layer_output[0, :, 0]
    max_ind = np.argmax(first_filter)

    text = linereader.readline(text_line)\
        .split(';')[1]\
        .replace('$NL$', '\n')\
        .replace('$SC$', ';')

    with open('outfile.html', 'w') as html_file:
        html_file.write('<html><body><table border-spacing=0 cellpadding=0 ' +
                        'cellspacing=0 table-layout=fixed>')

        splits = np.arange(80, first_filter.shape[0], 80)
        for i, line in enumerate(np.array_split(first_filter, splits)):
            html_file.write('<tr>')
            for j, value in enumerate(line):
                index = i * 80 + j

                if index >= len(text):
                    char = '_'
                else:
                    char = text[index]

                if index >= max_ind and index < max_ind + conv_size:
                    html_file.write('<td bgcolor="red">{}</td>'.format(char))
                else:
                    html_file.write('<td bgcolor="white">{}</td>'.format(char))
            html_file.write('</tr>')

        html_file.write('</table></body></html>')

    for i in range(1, 500):
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
