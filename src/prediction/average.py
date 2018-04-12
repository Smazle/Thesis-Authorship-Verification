#!/usr/bin/python3

import numpy as np
from keras.models import load_model
from ..preprocessing import LineReader, MacomReader
import argparse
import jsonpickle
import random

parser = argparse.ArgumentParser(
    description='Use neural network to predict authorship of assignments.'
)
parser.add_argument(
    'network',
    type=str,
    help='Path to file containing network we should use to predict.'
)
parser.add_argument(
    'reader',
    type=str,
    help='Path to file containing a macomreader.'
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to file containing the texts we work with.'
)
args = parser.parse_args()

# Load the keras model and the data reader.
model = load_model(args.network)
with open(args.reader, mode='r') as reader_in:
    reader = jsonpickle.decode(reader_in.read())

validation_set_reader = MacomReader(args.datafile, validation_split=1.0)

# For each author take 1 author text and 1 other text.
problems = []
for author in validation_set_reader.authors:
    other = list(reader.authors.keys())
    other.remove(author)

    author_texts = reader.authors[author]

    chosen_text_author = random.choice(author_texts)
    chosen_text_other = random.choice(reader.authors[random.choice(other)])

    author_texts.remove(chosen_text_author)

    problems.append((chosen_text_author, author_texts, True))
    problems.append((chosen_text_other, author_texts, False))

# Try to predict each problem.
tps = 0
tns = 0
fps = 0
fns = 0
with LineReader(args.datafile, encoding='utf-8') as linereader:
    for i, (unknown, knowns, label) in enumerate(problems):
        print('problem', i, 'label', label)
        unknown_text = np.zeros((1, reader.max_len))
        unknown_text[0] = reader.read_encoded_line(linereader, unknown)

        predictions = []

        for known in knowns:
            known_text = np.zeros((1, reader.max_len))
            known_text[0] = reader.read_encoded_line(linereader, known)

            prediction = model.predict([unknown_text, known_text],
                                       batch_size=1)

            predictions.append(prediction[0][1])

            print('\t', prediction[0][1])

        final_prediction = np.mean(predictions) > 0.5

        print(final_prediction)

        if final_prediction == label and label == True:
            tps = tps + 1
        elif final_prediction == label and label == False:
            tns = tns + 1
        elif final_prediction != label and label == True:
            fns = fns + 1
        elif final_prediction != label and label == False:
            fps = fps + 1
        else:
            raise Exception('This case should be impossible')

        print('tps,tns,fps,fns')
        print(tps, tns, fps, fns)

print('tps,tns,fps,fns')
print(tps, tns, fps, fns)
