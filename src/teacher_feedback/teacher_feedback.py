#!/usr/bin/env python3

import argparse
from keras import backend as K
import jsonpickle
import keras.models as M
import numpy as np
import src.preprocessing.macomreader as Macom
import src.util.utilities as util


def main():
    args = parse_arguments()
    model = M.load_model(args.model)
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())
        # Our reader should use the datafile we are given.
        reader.filepath = args.datafile,
        reader.batch_size = 1
        reader.authors = {}

    with Macom.LineReader(args.datafile) as linereader:
        # Generate the authors the reader uses.
        reader.authors = reader.generate_authors(linereader)

        assert args.author in reader.authors
        assert args.text < len(linereader.line_offsets)

        for candidate in reader.authors[args.author]:
            print('CANDIDATE AUTHORS TEXT', candidate)
            compare_texts(
                model, reader, linereader, candidate, args.text, args.n_most)


def compare_texts(model, reader, linereader, text1, text2, n):
    text1_channels = util.add_dim_start_all(
        reader.read_encoded_line(linereader, text1))
    text2_channels = util.add_dim_start_all(
        reader.read_encoded_line(linereader, text2))

    text1 = read_clean(linereader, text1)
    text2 = read_clean(linereader, text2)

    result = find_difference(model, text1_channels, text2_channels, n)
    results = zip(result['filter_index'], result['text_1_character_index'],
                  result['text_2_character_index'], result['text_1_value'],
                  result['text_2_value'])
    for filter_index, text1_ind, text2_ind, text1_val, text2_val in results:
        part1 = text1[text1_ind:text1_ind + 8]
        part2 = text2[text2_ind:text2_ind + 8]

        print('\tDifference in filter', filter_index)
        print('\t', repr(part1), text1_val)
        print('\t', repr(part2), text2_val)


def find_difference(model, text1_channels, text2_channels, n):
    layer_output_f = get_output_of_layer(model, 'convolutional_8')
    [text1_out, text2_out] = layer_output_f(
        text1_channels + text2_channels + [0])

    # Each row in text1_out and text2_out corresponds to the output of a
    # particular filter in the convolutional layer. Each column correspond to a
    # different index in the text.
    text1_out, text2_out = text1_out[0].T, text2_out[0].T

    # We should have same number of filters in each.
    assert text1_out.shape[0] == text2_out.shape[0]

    text1_max = np.amax(text1_out, axis=1)
    text2_max = np.amax(text2_out, axis=1)

    # The n greatest filter differences.
    n_greatest = np.argsort(np.abs(text1_max - text2_max))[::-1][0:n]

    # The index of the maximum value of the n max filters.
    text1_ind = np.argmax(text1_out[n_greatest], axis=1)
    text2_ind = np.argmax(text2_out[n_greatest], axis=1)

    # The value of the maximum value of the n max filters.
    text1_value = text1_out[n_greatest, text1_ind]
    text2_value = text2_out[n_greatest, text2_ind]

    return {
        'filter_index': n_greatest,
        'text_1_character_index': text1_ind,
        'text_2_character_index': text2_ind,
        'text_1_value': text1_value,
        'text_2_value': text2_value
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        'Gives feedback to teachers about why an assignment might be ghost '
        'written. The script takes an author id and a line number. The text '
        'on the line number is then compared to each of the authors texts. '
        'For each of the comparisons the script reports the --most-n most '
        'different feature values. The assumption is then that the largest '
        'differences is also the most important differences. An example '
        'invocation would be,\n'
        '\n'
        '\tpython3 src.teacher_feedback.teacher_feedback ./datafile.csv '
        './model.hdf5 ./reader.p 12345 777 --n-most=10\n'
        '\n'
        'That invocation would compare the text on line 777 with each of '
        'author 12345\'s texts and report the 10 largest feature differences '
        'for each of the texts.'
    )
    parser.add_argument(
        'datafile',
        type=str,
        help='File containing set of authors.'
    )
    parser.add_argument(
        'model',
        type=str,
        help='A trained instance of conv-char-NN (network3) to use for '
             'prediction.'
    )
    parser.add_argument(
        'reader',
        type=str,
        help='An instance of the macomreader that can be used to read texts.'
    )
    parser.add_argument(
        'author',
        type=str,
        help='The id of the candidate author.'
    )
    parser.add_argument(
        'text',
        type=int,
        help='The line number of the text we are verifying authorship of.'
    )
    parser.add_argument(
        '--n-most',
        type=int,
        default=1,
        help='How many of the most divergent feature activations to provide '
             'as output.'
    )
    args = parser.parse_args()

    return args


def get_output_of_layer(model, layer_name):
    return K.function([
        model.get_layer('known_input').input,
        model.get_layer('unknown_input').input,
        K.learning_phase()
    ], [
        model.get_layer(layer_name).get_output_at(0),
        model.get_layer(layer_name).get_output_at(1)
    ])


def read_clean(linereader, line):
    _, _, text = linereader.readline(line).split(';')
    return util.clean(text)


if __name__ == '__main__':
    main()
