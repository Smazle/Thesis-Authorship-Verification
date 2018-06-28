#!/usr/bin/env python3

import argparse
from keras import backend as K
import jsonpickle
import keras.models as M
import numpy as np
import src.preprocessing.macomreader as Macom
import src.util.utilities as util


# Represent the n largest differences in filters between two texts.
# filter_number is a list of indices of the filters with the largest
# differences, text1_max_index is the offset in the text where the largest
# value were found for the first text, text2_max_index is similar for the
# second text, text1_value is the max value obtained for the first text,
# text2_value is similar for the second text and size is the size of the
# convolutional window.
class FeatureMaxDifference:

    def __init__(self, filter_number, text1_max_index, text2_max_index,
                 text1_value, text2_value, size):

        self.filter_number = filter_number
        self.text1_max_index = text1_max_index
        self.text2_max_index = text2_max_index
        self.text1_value = text1_value
        self.text2_value = text2_value
        self.size = size


# Showable result class. Used to prettyprint FeatureMaxDifference by providing
# the texts the differences are between.
class Result:

    def __init__(self, feature_max_difference, text1, text2):
        self.feature_max_difference = feature_max_difference
        self.text1 = text1
        self.text2 = text2

        text1_max_index = self.feature_max_difference.text1_max_index
        text2_max_index = self.feature_max_difference.text2_max_index
        size = self.feature_max_difference.size

        self.text1_char_n_grams = []
        self.text2_char_n_grams = []
        self.text1_value = self.feature_max_difference.text1_value
        self.text2_value = self.feature_max_difference.text2_value

        for ind1, ind2 in zip(text1_max_index, text2_max_index):
            self.text1_char_n_grams.append(text1[ind1:ind1 + size])
            self.text2_char_n_grams.append(text2[ind2:ind2 + size])
            # self.text1_value.append(featurehjmcclhj)

    def __str__(self):
        filter_inds = self.feature_max_difference.filter_number
        text1s = self.text1_char_n_grams
        text2s = self.text2_char_n_grams
        value1s = self.text1_value
        value2s = self.text2_value

        zipped = zip(filter_inds, text1s, text2s, value1s, value2s)

        string = ''
        for filter_ind, text1, text2, value1, value2 in zipped:
            string += '\tfilter {}\ttext1 {}\ttext2 {}\tvalue1 {:^.2f}\tvalue2 {:^.2f}\n'\
                .format(filter_ind, repr(text1), repr(text2), value1, value2)

        return string


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

    for r in result:
        print(Result(r, text1, text2))


def find_difference(model, text1_channels, text2_channels, n):
    # Construct functions to get outputs from feature extraction layers.
    output8_1 = get_output_of_layer(model, 'convolutional_8')
    output8_2 = get_output_of_layer(model, 'convolutional_8_2')
    output4 = get_output_of_layer(model, 'convolutional_4')

    # Extract output.
    [text1_out8_1, text2_out8_1] = util.remove_dim_start_all(output8_1(
        text1_channels + text2_channels + [0]))
    [text1_out8_2, text2_out8_2] = util.remove_dim_start_all(output8_2(
        text1_channels + text2_channels + [0]))
    [text1_out4, text2_out4] = util.remove_dim_start_all(output4(
        text1_channels + text2_channels + [0]))

    # Each row in text1_out and text2_out corresponds to the output of a
    # particular filter in the convolutional layer. Each column correspond to a
    # different index in the text.
    text1_out8 = np.vstack([text1_out8_1.T, text1_out8_2.T])
    text2_out8 = np.vstack([text2_out8_1.T, text2_out8_2.T])
    text1_out4 = text1_out4.T
    text2_out4 = text2_out4.T

    # We should have same number of filters in each.
    assert text1_out8.shape[0] == text2_out8.shape[0]
    assert text1_out4.shape[0] == text2_out4.shape[0]

    text1_max8 = np.amax(text1_out8, axis=1)
    text2_max8 = np.amax(text2_out8, axis=1)
    text1_max4 = np.amax(text1_out4, axis=1)
    text2_max4 = np.amax(text2_out4, axis=1)

    # The n greatest filter differences.
    n_greatest8 = np.argsort(np.abs(text1_max8 - text2_max8))[::-1][0:n]
    n_greatest4 = np.argsort(np.abs(text1_max4 - text2_max4))[::-1][0:n]

    # The index of the maximum value of the n max filters.
    text1_ind8 = np.argmax(text1_out8[n_greatest8], axis=1)
    text2_ind8 = np.argmax(text2_out8[n_greatest8], axis=1)
    text1_ind4 = np.argmax(text1_out4[n_greatest4], axis=1)
    text2_ind4 = np.argmax(text2_out4[n_greatest4], axis=1)

    # The value of the maximum value of the n max filters.
    text1_value8 = text1_out8[n_greatest8, text1_ind8]
    text2_value8 = text2_out8[n_greatest8, text2_ind8]
    text1_value4 = text1_out4[n_greatest4, text1_ind4]
    text2_value4 = text2_out4[n_greatest4, text2_ind4]

    return [
        FeatureMaxDifference(n_greatest8, text1_ind8, text2_ind8,
                             text1_value8, text2_value8, 8),
        FeatureMaxDifference(n_greatest4, text1_ind4, text2_ind4,
                             text1_value4, text2_value4, 4)
    ]


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
