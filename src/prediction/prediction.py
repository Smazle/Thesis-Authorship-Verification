#!/usr/bin/env python3

import numpy as np
from keras.models import load_model
from ..preprocessing import LineReader, MacomReader
import argparse
import itertools
import jsonpickle
import random


SECS_PER_MONTH = 60 * 60 * 24 * 30


def get_problems(macomreader, linereader, negative_chance=1.0):
    problems = []
    for author in macomreader.authors:
        other = list(macomreader.authors.keys())
        other.remove(author)

        author_texts = macomreader.authors[author]

        if len(author_texts) <= 1:
            print("WARNING NOT ENOUGH TEXTS FOUND")
            continue

        # We want to predict the newest text.
        lines = [
            macomreader.read_encoded_line(linereader, line, with_date=True)
            for line in author_texts
        ]
        times = [time for _, time in lines]
        chosen_text_author = author_texts[np.argmax(times)]
        chosen_text_other = random.choice(
            macomreader.authors[random.choice(other)])

        author_texts.remove(chosen_text_author)

        problems.append((chosen_text_author, author_texts, True))

        if random.random() <= negative_chance:
            problems.append((chosen_text_other, author_texts, False))

    return problems


def predict(macomreader, linereader, author_texts, non_author_text, w, theta):
    unknown_text = macomreader.read_encoded_line(linereader, non_author_text)
    unknown_text = unknown_text.reshape((1, unknown_text.shape[0]))
    times = np.zeros((len(author_texts)), dtype=np.int)
    predictions = np.zeros((len(author_texts), ), dtype=np.float)

    # Read texts.
    for i, known in enumerate(author_texts):
        known_text, times[i] = macomreader.read_encoded_line(
            linereader, known, with_date=True)
        known_text = known_text.reshape((1, known_text.shape[0]))
        predictions[i] = model.predict([unknown_text, known_text])[0, 1]

    times = np.around((np.max(times) - times) / SECS_PER_MONTH)
    weights = exp_norm(times, w)

    return np.average(predictions, weights=weights) > theta


def evaluate(macomreader, linereader, problems, w, theta):
    tps, tns, fps, fns = 0, 0, 0, 0
    for i, (unknown, knowns, label) in enumerate(problems):
        prediction = predict(
            macomreader, linereader, knowns, unknown, w, theta)

        if prediction == label and label == True:
            tps = tps + 1
        elif prediction == label and label == False:
            tns = tns + 1
        elif prediction != label and label == True:
            fns = fns + 1
        elif prediction != label and label == False:
            fps = fps + 1
        else:
            raise Exception('This case should be impossible')

    return tps, tns, fps, fns


def exp_norm(xs, l):
    xs = np.exp(-l * xs)
    return xs / np.sum(xs)


if __name__ == '__main__':
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
    parser.add_argument(
        '--theta',
        nargs='+',
        help='Thresholds to use.',
        default=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        help='The argument given to the exponential dropoff weight ' +
             'function. If 0.0 is given it is equivalent to uniform weights.',
        default=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    )
    parser.add_argument(
        '--negative-chance',
        help='The fraction of negative problems wanted.',
        default=1.0,
        type=float
    )
    args = parser.parse_args()

    theta = list(map(lambda x: float(x), args.theta))
    weights = list(map(lambda x: float(x), args.weights))

    # Load the keras model and the data reader.
    model = load_model(args.network)
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())

    assert reader.vocabulary_map is not None
    assert reader.padding is not None
    assert reader.char is not None
    assert reader.garbage is not None
    assert reader.max_len is not None
    assert reader.pad is not None
    assert reader.binary is not None
    assert reader.batch_normalization is not None

    validation_reader = MacomReader(
        args.datafile,
        char=reader.char,
        validation_split=1.0,
        batch_size=1,
        pad=reader.pad,
        binary=reader.binary,
        batch_normalization=reader.batch_normalization
    )
    validation_reader.vocabulary_map = reader.vocabulary_map
    validation_reader.padding = reader.padding
    validation_reader.garbage = reader.garbage
    validation_reader.max_len = reader.max_len

    with LineReader(args.datafile) as linereader:
        problems = get_problems(validation_reader, linereader,
                                negative_chance=args.negative_chance)

        positive_n = sum([1 for (_, _, label) in problems if label])
        negative_n = sum([1 for (_, _, label) in problems if not label])

        print('Generated {} positives and {} negatives'
              .format(positive_n, negative_n))

        print('Theta,Weights,TPS,TNS,FPS,FNS,ACC,ERR', end='\r\n')
        for (theta, weight) in itertools.product(theta, weights):
            tps, tns, fps, fns = evaluate(
                validation_reader, linereader, problems, weight, theta)

            accuracy = (tps + tns) / (tps + tns + fps + fns)
            if fns + tns == 0:
                errors = 0
            else:
                errors = fns / (fns + tns)

            print('{},{},{},{},{},{},{},{}'.format(
                theta, weight, tps, tns, fps, fns, accuracy, errors),
                end='\r\n')