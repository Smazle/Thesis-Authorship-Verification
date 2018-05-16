# -*- coding: utf-8 -*-
# !/usr/bin/python3

from .channels import ChannelType, CharVocabulary, WordVocabulary, \
        SentenceVocabulary, vocabulary_factory
import itertools
import numpy as np
import pickle
import random
import sys
from ..util import utilities as util
from datetime import datetime
from keras.preprocessing import sequence


class LineReader:

    # Name of the file we read lines from.
    filename = None

    # List of line offsets.
    line_offsets = None

    # Format to read lines as.
    encoding = None

    # File handle to read from.
    f = None

    def __init__(self, filename, encoding='utf-8'):
        self.filename = filename
        self.encoding = encoding

        self.line_offsets = []
        with open(self.filename, mode='rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)

    def __enter__(self):
        self.f = open(self.filename, mode='r', encoding=self.encoding)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

    def readline(self, line_n):
        self.f.seek(self.line_offsets[line_n])

        return self.f.readline()

    def readlines(self, skipfirst=False):
        if skipfirst:
            for i in range(1, len(self.line_offsets)):
                yield self.readline(i)
        else:
            for i in range(0, len(self.line_offsets)):
                yield self.readline(i)


class MacomReader(object):

    # The maximal length of any of the texts.
    max_len = 0

    # Number between 0 and 1 (inclusive) cutoff point for when a character is
    # replaced with the default character.
    vocabulary_frequency_cutoff = 0.0

    # The path of the datafile.
    filepath = None

    # The size of each batch we yield in the generator.
    batch_size = None

    # Map from author identifier to list of line numbers.
    authors = {}

    # List of problems which consist of two line indices to two texts and
    # either 1 or 0. If 1 the texts are from the same author and if 0 they are
    # from different authors.
    problems = []

    # If 0..8 80% is training data and 20% are validation data.
    validation_split = None

    # List of training problems.
    training_problems = None

    # List of validation problems.
    validation_problems = None

    # Whether or not to pad the texts with a special value.
    pad = None

    # Whether to return y as binary or categorical crossentropy.
    binary = None

    # TODO: Take argument specifying whether or not to ignore first line in
    # file.
    def __init__(self, filepath, batch_size=32, validation_split=0.8,
                 vocabulary_frequency_cutoff=0.0, pad=True, binary=False,
                 batch_normalization='truncate', channels=[ChannelType.CHAR]):

        if validation_split > 1.0 or validation_split < 0.0:
            raise ValueError('validation_split between 0 and 1 required')

        if batch_normalization not in ['truncate', 'pad']:
            raise ValueError('Only truncate and pad is currently supported.')

        for channel in channels:
            if channel not in [ChannelType.CHAR, ChannelType.WORD,
                    ChannelType.SENTENCE]:
                raise ValueError('Only char, word or sentence channels allowed')

        # Save parameters.
        self.filepath = filepath
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff # TODO: Should be list.
        self.binary = binary
        self.pad = pad
        self.batch_normalization = batch_normalization
        self.channeltypes = channels

        if self.binary:
            self.label_true = np.array([1])
            self.label_false = np.array([0])
        else:
            self.label_true = np.array([0, 1]).reshape(1, 2)
            self.label_false = np.array([1, 0]).reshape(1, 2)

        # Generate representation used to generate training data.
        with LineReader(self.filepath) as linereader:
            self.generate_authors(linereader)
            self.generate_problems()
            self.generate_vocabulary_maps(linereader)

    def generate_training(self):
        return self.generate(self.training_problems)

    def generate_validation(self):
        return self.generate(self.validation_problems)

    def generate_vocabulary_maps(self, linereader):
        def linegen():
            for author in self.authors:
                for line_n in self.authors[author]:
                    autor, date, text = linereader.readline(line_n).split(';')
                    decoded = util.clean(text)
                    yield decoded

        self.channels = []
        for channeltype in self.channeltypes:
            self.channels.append(vocabulary_factory(channeltype,
                                 self.vocabulary_frequency_cutoff, linegen()))

    def generate_authors(self, linereader):
        for i, line in enumerate(linereader.readlines(skipfirst=True)):
            author, date, text = line.split(';')
            text = util.clean(text)

            if len(text) > 30000:
                print('WARNING: Skipping text longer than 30,000 characters ' +
                      'on line {}'.format(i + 1))
            elif len(text) < 400:
                print('WARNING: Skipping text shorter than 400 characters ' +
                      'on line {}'.format(i + 1))
            elif author in self.authors:
                self.authors[author].append(i + 1)
            else:
                self.authors[author] = [i + 1]

    def generate_problems(self):

        for author in self.authors:
            other = set(self.authors.keys())
            other.remove(author)

            # Generate all combinations of the authors texts.
            for (l1, l2) in itertools.combinations(self.authors[author], r=2):
                # Generate a sample with same author.
                self.problems.append((l1, l2, 1))

                # Generate a sample with different authors.
                same = random.choice(self.authors[author])
                different = random.choice(self.authors
                                          [random.choice(list(other))])
                self.problems.append((same, different, 0))

        random.shuffle(self.problems)

        split_point = int(len(self.problems) * self.validation_split)
        self.training_problems = self.problems[:split_point]
        self.validation_problems = self.problems[split_point:]

    # Returns a list of numpy arrays containing integers where each array is an
    # encoded sequence. The list ordering corresponds to the self.channels
    # parameter.
    def read_encoded_line(self, linereader, line_n, with_date=False):
        if self.pad:
            raise Exception('read_encoded_line does not currently work with global padding')

        author, date, text = linereader.readline(line_n).split(';')
        unescaped = util.clean(text)
        unescaped = unescaped[200:]

        encoded_channels = []
        for channel in self.channels:
            encoded_channels.append(channel.encode(unescaped))

        if with_date:
            epoch = datetime.utcfromtimestamp(0)
            date = datetime.strptime(date, '%d-%m-%Y')
            time = (date - epoch).total_seconds()
            return encoded_channels, time
        else:
            return encoded_channels

    # Generate batches of samples.
    def generate(self, problems):
        with LineReader(self.filepath, encoding='utf-8') as reader:
            problems = itertools.cycle(problems)

            while True:
                batch = list(itertools.islice(problems, self.batch_size))
                known_inputs, unknown_inputs, y = self.generate_batch(batch, reader)
                yield known_inputs + unknown_inputs, y

    # TODO: Refactor the function. It looks like shit.
    def generate_batch(self, batch, linereader):
        knowns = []
        unknowns = []

        if self.binary:
            y = np.zeros((self.batch_size, 1))
        else:
            y = np.zeros((self.batch_size, 2))

        for (i, (known, unknown, label)) in enumerate(batch):
            knowns.append(self.read_encoded_line(linereader, known))
            unknowns.append(self.read_encoded_line(linereader, unknown))

            if label == 0:
                y[i] = self.label_false
            else:
                y[i] = self.label_true

        X_knowns = []
        X_unknowns = []
        if self.batch_normalization == 'truncate':
            for i in range(len(self.channels)):
                known_channel = list(map(lambda x: x[i], knowns))
                known_truncate_len = min(map(lambda x: x.shape[0], known_channel))
                unknown_channel = list(map(lambda x: x[i], unknowns))
                unknown_truncate_len = min(map(lambda x: x.shape[0], unknown_channel))

                X_known = sequence.pad_sequences(
                    known_channel,
                    value=self.padding,
                    maxlen=known_truncate_len,
                    truncating='post')
                X_unknown = sequence.pad_sequences(
                    unknown_channel,
                    value=self.padding,
                    maxlen=unknown_truncate_len,
                    truncating='post')

                X_knowns.append(X_known)
                X_unknowns.append(X_unknown)
        elif self.batch_normalization == 'pad':
            for i, channel in enumerate(self.channels):
                known_channel = list(map(lambda x: x[i], knowns))
                unknown_channel = list(map(lambda x: x[i], unknowns))
                X_known = sequence.pad_sequences(
                    known_channel, value=channel.padding, padding='post')
                X_unknown = sequence.pad_sequences(
                    unknown_channel, value=channel.padding, padding='post')

                X_knowns.append(X_known)
                X_unknowns.append(X_unknown)
        else:
            raise Exception('should never happen')

        return X_knowns, X_unknowns, y


if __name__ == '__main__':
    reader = MacomReader(
        sys.argv[1],
        vocabulary_frequency_cutoff=1 / 100000,
        validation_split=0.95,
        batch_size=2,
        pad=False,
        batch_normalization='pad',
        channels=[ChannelType.CHAR, ChannelType.CHAR, ChannelType.WORD]
    )

    for _, (inputs, label) in zip(range(6), reader.generate_training()):
        print(inputs)
        print(label)
        print(inputs[0].shape)
        print(inputs[1].shape)
        print(inputs[2].shape)
        print(inputs[3].shape)
