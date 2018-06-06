#!/usr/bin/env python3 # -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from ..preprocessing import LineReader, MacomReader
import argparse
import itertools
import jsonpickle
import random
import sys
from src.prediction.weight_functions import weight_factory
import matplotlib.pyplot as plt


def get_problems(macomreader, linereader, negative_chance=1.0):
    problems = []
    for author in macomreader.authors:
        other = list(macomreader.authors.keys())
        other.remove(author)

        author_texts = macomreader.authors[author]

        if len(author_texts) <= 1:
            print('WARNING NOT ENOUGH TEXTS FOUND', file=sys.stderr)
            continue

        # We want to predict the newest text.
        lines = [
            macomreader.read_encoded_line(linereader, line, with_meta=True)
            for line in author_texts
        ]
        times = [time for _, time, _ in lines]
        chosen_text_author = author_texts[np.argmax(times)]
        chosen_text_other = random.choice(
            macomreader.authors[random.choice(other)])

        author_texts.remove(chosen_text_author)

        problems.append((chosen_text_author, author_texts, True))

        if random.random() <= negative_chance:
            problems.append((chosen_text_other, author_texts, False))

    return problems


def predict(macomreader, linereader, author_texts, non_author_text):
    unknown_text = macomreader.read_encoded_line(linereader, non_author_text)
    unknown_text = list(map(lambda x: add_dim_start(x), unknown_text))
    times = np.zeros((len(author_texts)), dtype=np.int)
    predictions = np.zeros((len(author_texts), ), dtype=np.float)
    text_lengths = np.zeros((len(author_texts), ), dtype=np.int)

    # Read texts.
    for i, known in enumerate(author_texts):
        known_text, times[i], text_lengths[i] = macomreader.read_encoded_line(
            linereader, known, with_meta=True)
        known_text = list(map(lambda x: add_dim_start(x), known_text))
        predictions[i] = model.predict(known_text + unknown_text)[0, 1]

    return predictions, times, text_lengths


def predict_all(macomreader, linereader, problems):
    results = []

    for idx, (unknown, knowns, label) in enumerate(problems):
        print(idx, len(problems), file=sys.stderr)
        predictions, times, lengths = predict(macomreader, linereader, knowns, unknown)
        results.append((predictions, times, lengths))

    return results


def evaluate(labels, results, w, theta):
    tps, tns, fps, fns = 0, 0, 0, 0
    for label, (predictions, times, text_lengths) in zip(labels, results):
        weights = w.get_weights(times, predictions, text_lengths)
        prediction = np.average(predictions, weights=weights) > theta

        if prediction == label and label == True:  # noqa
            tps = tps + 1
        elif prediction == label and label == False:  # noqa
            tns = tns + 1
        elif prediction != label and label == True:  # noqa
            fns = fns + 1
        elif prediction != label and label == False:  # noqa
            fps = fps + 1
        else:
            raise Exception('This case should be impossible')

    return tps, tns, fps, fns


def add_dim_start(array):
    return np.reshape(array, [1] + list(array.shape))


def generate_graphs(weights, labels, results):
    thetas = np.linspace(0, 1, num=1000)

    for weight in weights:
        accuracies = []
        errors = []
        for theta in thetas:
            # Compute results.
            tps, tns, fps, fns = evaluate(labels, results, weight, theta)

            accuracy = (tps + tns) / (tps + tns + fps + fns)

            if fns + tns == 0:
                error = 0
            else:
                error = fns / (fns + tns)

            accuracies.append(accuracy)
            errors.append(error)


        label = 'Weight {}'.format(weight)
        plt.figure(1)
        plt.plot(thetas, accuracies, label=label)
        plt.figure(2)
        plt.plot(thetas, errors, label=label)

    plt.figure(1)
    plt.xlabel('Threshold (Theta)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.figure(2)
    plt.xlabel('Threshold (Theta)')
    plt.ylabel('Accusation Error')
    plt.grid(True)
    plt.legend()

    plt.show()


def binary_theta_search(weights, labels, results):
    limit_theta = 1
    lower_theta = 0
    print('\nStarting Fine tuned run', file=sys.stderr)
    print('{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}'
          .format(
              'L-Theta', 'U-Theta', 'A-Theta', 'Err', 'Acc', 'TNS',
              'FNS', 'TPS', 'FPS', 'Weight'))

    for _ in range(100):
        new_theta = (limit_theta + lower_theta) / 2
        e = [evaluate(labels, results, weight, new_theta)
             for weight in weights]

        accuracies = [((tns, fns, fps, tps),
                       (tps + tns) / (tps + tns + fps + fns))
                      for tps, tns, fps, fns in e]

        ((tns, fns, fps, tps), acc) = max(accuracies, key=lambda x: x[1])
        w = weights[accuracies.index(((tns, fns, fps, tps), acc))]

        if fns + tns == 0:
            errors = 0
        else:
            errors = fns / (fns + tns)

        if errors < 0.1:
            print(('\033[92m' +
                   '{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}' +
                   '{:^10}{:^10}{:^10}{:^10}{:^10}\033[0m')
                  .format(lower_theta, limit_theta, new_theta, errors, acc,
                          tns, fns, tps, fps, w))
            lower_theta = new_theta
        else:
            print(('{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}' +
                   '{:^10}{:^10}{:^10}{:^10}{:^10}')
                  .format(lower_theta, limit_theta, new_theta, errors, acc,
                          tns, fns, tps, fps, w))
            limit_theta = new_theta


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
        '--weights',
        nargs='+',
        help='The argument given to the exponential dropoff weight ' +
             'function. If 0.0 is given it is equivalent to uniform weights.',
        default=['exp-norm', 'maximum', 'minimum']
    )
    parser.add_argument(
        '--negative-chance',
        help='The fraction of negative problems wanted.',
        default=1.0,
        type=float
    )

    args = parser.parse_args()

    weights = sum([weight_factory(w) for w in args.weights], [])

    # Load the keras model and the data reader.
    model = load_model(args.network)
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())

    # Our reader should use the validation file we are given.
    reader.filepath = args.datafile,
    reader.validation_split = 1.0
    reader.batch_size = 1
    reader.authors = {}

    with LineReader(args.datafile) as linereader:
        # We have to generate new authors since we are probably using a new
        # dataset.
        reader.authors = reader.generate_authors(linereader)

        problems = get_problems(reader, linereader,
                                negative_chance=args.negative_chance)

        labels = [label for (_, _, label) in problems]
        positive_n = len(list(filter(lambda x: x, labels)))
        negative_n = len(list(filter(lambda x: not x, labels)))

        print('Generated {} positives and {} negatives'
              .format(positive_n, negative_n), file=sys.stderr)

        results = predict_all(reader, linereader, problems)

        generate_graphs(weights, labels, results)
        binary_theta_search(weights, labels, results)
