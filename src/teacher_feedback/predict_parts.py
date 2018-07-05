#!/usr/bin/env python3

import argparse
from keras import backend as K
import jsonpickle
import keras.models as M
import numpy as np
import src.preprocessing.macomreader as Macom
import src.util.utilities as util


def main():
    args = parse_args()
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

        text = util.read_clean(linereader, args.text)
        paragraphs = list(filter(lambda x: len(x) > 100, text.split('\n\n')))

        predictions = np.zeros((len(reader.authors[args.author]), len(paragraphs)))
        for i, candidate in enumerate(reader.authors[args.author]):
            print('HANDLING CANDIDATE TEXT {}'.format(candidate))

            # Read the candidate text.
            candidate_text = reader.read_encoded_line(linereader, candidate)[0]
            candidate_text = util.add_dim_start(candidate_text)

            for j, paragraph in enumerate(paragraphs):
                print(paragraph)
                encoded_paragraph = reader.channels[0].encode(paragraph)
                encoded_paragraph = util.add_dim_start(encoded_paragraph)
                prediction = model.predict([candidate_text, encoded_paragraph])[0,1]
                print('\t', prediction)

                predictions[i, j] = prediction

        print(predictions)
        print(np.mean(predictions, axis=0))


def parse_args():
    parser = argparse.ArgumentParser(
        'Predict each paragraph in a text independently to locate parts of '
        'texts that looks ghost written. The score given to each paragraph '
        'is the mean of the prediction score between each of the candidate '
        'authors texts. An example invocation of this script would be,'
        '\n'
        '\tpython3 src.teacher_feedback.predict_parts ./datafile.csv '
        './model.hdf5 ./reader.p 12345 777\n'
        '\n'
        'That invocation would compare the text on line 777 with each of '
        'author 12345\'s texts and report the mean score in each of the '
        'paragraphs in text 777.'
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
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
