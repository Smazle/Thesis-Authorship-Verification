#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from ..preprocessing import LineReader
import argparse
import time
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


def predict(model, macomreader, linereader, author_texts, non_author_text):
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


def predict_all(model, macomreader, linereader, problems):
    results = []

    for idx, (unknown, knowns, label) in enumerate(problems):
        print(idx, len(problems), file=sys.stderr)
        predictions, times, lengths = predict(model, macomreader, linereader,
                                              knowns, unknown)
        results.append((predictions, times, lengths))

    return results


def evaluate(labels, results, w, theta):
    tps, tns, fps, fns = 0, 0, 0, 0
    for label, (predictions, times, text_lengths) in zip(labels, results):
        prediction = w.predict(predictions, theta, times, text_lengths)

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
    total = len(weights)

    f, axarr = plt.subplots(2, sharex=True)

    for idx, weight in enumerate(weights):
        print('{}/{}'.format(idx, total), file=sys.stderr)
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

        label = str(weight).replace(', $\lambda$ = ', '')
        axarr[0].plot(thetas, accuracies, label=label)
        axarr[1].plot(thetas, errors, label=label)

    axarr[0].set_ylabel('Accuracy')
    axarr[0].grid(True)

    axarr[1].set_ylabel('Accusation Error')
    axarr[1].grid(True)
    axarr[1].legend()

    axarr[1].set_xlabel(r'$\theta$ (Threshold)')
    lgd = plt.legend(bbox_to_anchor=(1.25, 1), loc=7, fancybox=True)
    plt.show()
    f.savefig(
        'Prediction_{}.png'.format(time.time()),
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight')


def binary_theta_search(weights, labels, results):
    limit_theta = 1
    lower_theta = 0
    print('\nStarting Fine tuned run', file=sys.stderr)

    for i in np.linspace(0.1, 1, 10):
        print('{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}' +
              '{:^10}{:^10}{:^10}{:^10}{:^10}'
              .format('L-Theta', 'U-Theta', 'A-Theta', 'Err', 'Acc', 'TNS',
                      'FNS', 'TPS', 'FPS', 'Weight', 'Theta'))
        for _ in range(50):
            new_theta = (limit_theta + lower_theta) / 2
            e = [
                evaluate(labels, results, weight, new_theta)
                for weight in weights
            ]

            accuracies = [((tns, fns, fps, tps),
                           (tps + tns) / (tps + tns + fps + fns))
                          for tps, tns, fps, fns in e]

            ((tns, fns, fps, tps), acc) = max(accuracies, key=lambda x: x[1])
            w = weights[accuracies.index(((tns, fns, fps, tps), acc))]

            if fns + tns == 0:
                errors = 0
            else:
                errors = fns / (fns + tns)

            if errors < i:
                print(
                    ('\033[92m' + '{:^10.6f}{:^10.6f}{:^10.6f}' +
                     '{:^10.6f}{:^10.6f}' +
                     '{:^10}{:^10}{:^10}{:^10}{:^10}{:^10.1f}\033[0m').format(
                         lower_theta, limit_theta, new_theta, errors, acc, tns,
                         fns, tps, fps, str(w), i))
                lower_theta = new_theta
            else:
                print(('{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}{:^10.6f}' +
                       '{:^10}{:^10}{:^10}{:^10}{:^10}{:^10.1f}').format(
                           lower_theta, limit_theta, new_theta, errors, acc,
                           tns, fns, tps, fps, str(w), i))
                limit_theta = new_theta

        print('\n\n')


def main():
    parser = argparse.ArgumentParser(
        description='Use neural network to predict authorship of assignments.')
    parser.add_argument(
        'network',
        type=str,
        help='Path to file containing network we should use to predict.')
    parser.add_argument(
        'reader', type=str, help='Path to file containing a macomreader.')
    parser.add_argument(
        'datafile',
        type=str,
        help='Path to file containing the texts we work with.')
    parser.add_argument(
        '--weights',
        nargs='+',
        help='Which weights to use.',
        default=['exp-norm', 'maximum', 'minimum', 'majority-vote', 'text'])
    parser.add_argument(
        '--negative-chance',
        help='The fraction of negative problems wanted.',
        default=1.0,
        type=float)

    args = parser.parse_args()

    weights = sum([weight_factory(w) for w in args.weights], [])

    # Load the keras model and the data reader.
    model = load_model(args.network)
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())

    # Our reader should use the validation file we are given.
    reader.filepath = args.datafile,
    reader.batch_size = 1
    reader.authors = {}

    with LineReader(args.datafile) as linereader:
        # We have to generate new authors since we are probably using a new
        # dataset.
        reader.authors = reader.generate_authors(linereader)

        problems = get_problems(
            reader, linereader, negative_chance=args.negative_chance)

        labels = [label for (_, _, label) in problems]
        positive_n = len(list(filter(lambda x: x, labels)))
        negative_n = len(list(filter(lambda x: not x, labels)))

        print(
            'Generated {} positives and {} negatives'.format(
                positive_n, negative_n),
            file=sys.stderr)

        results = predict_all(model, reader, linereader, problems)

        generate_graphs(weights, labels, results)
        binary_theta_search(weights, labels, results)


if __name__ == '__main__':
    main()
