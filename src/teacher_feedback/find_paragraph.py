#!/usr/bin/python3

import argparse
import jsonpickle
import keras.models as M
import numpy as np
import random
import src.preprocessing.macomreader as Macom
import src.util.utilities as util
import sys


PARAGRAPH_DELIMITER = '\n\n'


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

        # Print CSV header.
        print('pos1,pos2,pos3,pos4,pos5,neg1', end='\r\n')

        for i, author in enumerate(reader.authors):
            info('Handling author {} ({}/{})'.format(author, i, len(reader.authors)))
            handle_author(model, reader, linereader, author)


def handle_author(model, macomreader, linereader, author):

    for text_index in macomreader.authors[author]:
        paragraphs = read_paragraphs(linereader, text_index)

        if len(paragraphs) >= 5:
            predict_paragraphs(model, macomreader, linereader, author, text_index)
            break
    else:
        warn('Author {} has no texts with enough paragraphs.'.format(author))


def predict_paragraphs(model, macomreader, linereader, author, text):
    author_texts = macomreader.authors[author]
    author_texts.remove(text)

    # Find a single paragraph from another text that we can use as negative.
    neg_paragraph = find_negative_paragraph(macomreader, linereader, author)
    encoded_neg_paragraph = macomreader.channels[0].encode(neg_paragraph)
    encoded_neg_paragraph = util.add_dim_start(encoded_neg_paragraph)

    # Take the first 5 paragraphs.
    paragraphs = read_paragraphs(linereader, text)
    encoded_paragraphs = list(map(lambda x: macomreader.channels[0].encode(x),
                              paragraphs))[0:5]
    encoded_paragraphs = list(map(lambda x: util.add_dim_start(x),
                              encoded_paragraphs))

    # For each of the authors texts predict each of the first 5 positive
    # paragraphs and the single negative paragraph.
    negative_predictions = np.zeros((len(author_texts), 1))
    positive_predictions = np.zeros((len(author_texts), 5))
    for i, author_text in enumerate(author_texts):
        encoded_text = macomreader.channels[0].encode(util.read_clean(linereader, author_text))
        encoded_text = util.add_dim_start(encoded_text)

        # Predict all the positive paragraphs.
        for j, encoded_paragraph in enumerate(encoded_paragraphs):
            prediction = model.predict([encoded_text, encoded_paragraph])[0, 1]
            positive_predictions[i, j] = prediction

        # Predict the single negative paragraph.
        prediction = model.predict([encoded_text, encoded_neg_paragraph])[0, 1]
        negative_predictions[i, 0] = prediction

    pos_average = np.average(positive_predictions, axis=0)
    neg_average = np.average(negative_predictions, axis=0)

    print('{},{},{},{},{},{}'.format(pos_average[0], pos_average[1],
                                     pos_average[2], pos_average[3],
                                     pos_average[4], neg_average[0]),
          end='\r\n')


def find_negative_paragraph(macomreader, linereader, author):
    other_authors = list(filter(lambda x: x != author, macomreader.authors))
    negative_author = random.choice(other_authors)
    negative = random.choice(macomreader.authors[negative_author])

    paragraphs = read_paragraphs(linereader, negative)

    if len(paragraphs) == 0:
        return find_negative_paragraph(macomreader, linereader, author)
    else:
        return random.choice(paragraphs)


# Read text and return all paragraphs longer than minimum_length.
def read_paragraphs(linereader, text, minimum_length=100):
    string = util.read_clean(linereader, text)
    paragraphs = string.split(PARAGRAPH_DELIMITER)
    paragraphs = list(filter(lambda x: len(x) > minimum_length, paragraphs))
    return paragraphs


def info(string):
    print('INFO: ' + string, file=sys.stderr)


def warn(string):
    print('WARN: ' + string, file=sys.stderr)


def error(string):
    print('ERROR: ' + string, file=sys.stderr)
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        'Locate the paragraph in a specified text that looks the least like a '
        'writers normal writing style. Currently we loop through all authors '
        'in the dataset and sample sentences from each of them. Paa sigt that '
        'should probably be done from another file and this file should just '
        'handle a single author and single text.'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to trained conv-char-NN model.'
    )
    parser.add_argument(
        'reader',
        type=str,
        help='Path to macomreader associated with the model.'
    )
    parser.add_argument(
        'datafile',
        type=str,
        help='Path to datafile to read from.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
