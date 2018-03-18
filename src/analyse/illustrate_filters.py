#!/usr/bin/env python3

import numpy as np
from keras.models import load_model
from keras import backend as K
from ..preprocessing import load_reader
import sys


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


model = load_model(sys.argv[1])

get_output = K.function([
    model.get_layer('known_input').input,
    model.get_layer('unknown_input').input,
    K.learning_phase()], [
    model.get_layer('convolutional_8').get_output_at(0)]
)

macomreader = load_reader(sys.argv[2])
macomreader.batch_size = 1

text_line = 10
opposition = 2

with macomreader as reader:
    text1 = macomreader.read_line(text_line, macomreader.f)
    text2 = macomreader.read_line(opposition, macomreader.f)

    text1 = np.expand_dims(text1, axis=0)
    text2 = np.expand_dims(text2, axis=0)

    layer_output = get_output([text1, text2, 0])[0]

    # Extract first filter.
    first_filter = layer_output[0, :, 0]
    minimum = np.amin(first_filter)
    maximum = np.amax(first_filter)

    macomreader.f.seek(macomreader.line_offset[text_line])
    text = macomreader.f.readline()\
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

                r, g, b = rgb(minimum, maximum, value)
                html_file.write('<td bgcolor="rgb({},{},{})"><font color="white">{}</font></td>'\
                    .format(r, g, b, char))
            html_file.write('</tr>')

        html_file.write('</table></body></html>')
