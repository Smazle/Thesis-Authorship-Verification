#!/usr/bin/env python3

import numpy as np
from keras.models import load_model
from ..preprocessing import LineReader, MacomReader
import argparse
import itertools
import jsonpickle
import random


def get_problems(macomreader, linereader):
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
        problems.append((chosen_text_other, author_texts, False))

    return problems


def predict(macomreader, linereader, author_texts, non_author_text, w, theta):
    unknown_text = np.zeros((1, macomreader.max_len), dtype=np.int)
    unknown_text[0] = macomreader.read_encoded_line(linereader, non_author_text)
    times = np.zeros((len(author_texts)), dtype=np.int)
    predictions = np.zeros((len(author_texts), ), dtype=np.float)

    # Read texts.
    for i, known in enumerate(author_texts):
        known_text = np.zeros((1, macomreader.max_len), dtype=np.int)
        known_text[0], times[i] = macomreader.read_encoded_line(
            linereader, known, with_date=True)
        predictions[i] = model.predict([unknown_text, known_text])[0, 1]

    return np.average(predictions, weights=w(times)) > theta


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


# Return uniform distribution over the timestamps.
def uniform(xs):
    return np.ones((xs.shape[0],), dtype=np.float) / xs.shape[0]


# Return a distribution over the times such that the newest time has weight
# 1/2, the second newest has 1/4 and so on.
def time_simple(xs):
    weights = [1 / (2**x) for x in range(1, len(xs))]
    if len(weights) == 0:
        weights = [1.0]
    else:
        weights.append(weights[-1])
    sort = np.flip(np.argsort(xs), 0)

    return np.array(weights)[sort]


# Weight by time in seconds such that the newest text is given the highest
# weight. The oldest text will have weight 0.
def time_weighted(xs):
    weights = xs - np.min(xs)
    weights = weights / np.sum(weights)

    return weights


# Weight by time monthly and arbitrarily add one to all of them to give the
# oldest text another weight than 0.
def time_weighted_2(xs):
    seconds_per_month = 2629743
    xs = xs / seconds_per_month

    weights = xs - np.min(xs) + 1
    weights = weights / np.sum(weights)

    return weights


def get_weight(weight_name):
    if args.weights == 'uniform':
        return uniform
    elif args.weights == 'simple-time':
        return time_simple
    elif args.weights == 'advanced-time':
        return time_weighted
    elif args.weights == 'advanced-time-2':
        return time_weighted_2
    else:
        raise Exception('Unknown weights {}'.format(args.weights))


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
        default=list(np.arange(0.0, 1.0, 0.1))
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        help='Which weighting of predictions to use. Should be one of ' +
             '"uniform", "simple-time", "advanced-time" or "advanced-time-2"',
        default=['uniform']
    )
    args = parser.parse_args()

    # Load the keras model and the data reader.
    model = load_model(args.network)
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())

    assert reader.vocabulary_map is not None
    assert reader.padding is not None
    assert reader.char is not None
    assert reader.garbage is not None
    assert reader.max_len is not None

    validation_reader = MacomReader(args.datafile, char=reader.char,
                                    validation_split=1.0)
    validation_reader.vocabulary_map = reader.vocabulary_map
    validation_reader.padding = reader.padding
    validation_reader.garbage = reader.garbage
    validation_reader.max_len = reader.max_len

    with LineReader(args.datafile) as linereader:
        problems = get_problems(validation_reader, linereader)

        print('Theta,Weights,TPS,TNS,FPS,FNS,ACC,ERR', end='\r\n')
        for (theta, weight) in itertools.product(args.theta, args.weights):
            w = get_weight(weight)

            tps, tns, fps, fns = evaluate(
                validation_reader, linereader, problems, w, theta)

            accuracy = (tps + tns) / (tps + tns + fps + fns)
            if fns + tns == 0:
                errors = 0
            else:
                errors = fns / (fns + tns)

            print('{},{},{},{},{},{},{},{}'.format(
                theta, weight, tps, tns, fps, fns, accuracy, errors),
                end='\r\n')
