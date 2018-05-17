#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .channels import ChannelType, vocabulary_factory
import itertools
import numpy as np
import random
from ..util import utilities as util
from datetime import datetime
from keras.preprocessing import sequence
from nltk.tokenize import sent_tokenize


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

    # If 0.8 80% is training data and 20% are validation data.
    validation_split = None

    # List of training problems.
    training_problems = None

    # List of validation problems.
    validation_problems = None

    # Whether or not to pad the texts with a special value.
    pad = None

    # Whether to return y as binary or categorical crossentropy.
    binary = None

    # Sentence length, in case of a sentence channel being used
    sentence_length = None

    def __init__(self, filepath, batch_size=32, validation_split=0.8,
                 vocabulary_frequency_cutoff=[0.0], pad=True, binary=False,
                 batch_normalization='truncate', channels=[ChannelType.CHAR],
                 sentence_len=None):

        if validation_split > 1.0 or validation_split < 0.0:
            raise ValueError('validation_split between 0 and 1 required')

        if batch_normalization not in ['truncate', 'pad']:
            raise ValueError('Only truncate and pad is currently supported.')

        for channel in channels:
            if channel not in [ChannelType.CHAR, ChannelType.WORD,
                               ChannelType.SENTENCE]:
                raise ValueError(
                    'Only char, word or sentence channels allowed')

                # Save parameters.

        self.filepath = filepath
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff
        self.binary = binary
        self.pad = pad
        self.batch_normalization = batch_normalization
        self.channeltypes = channels
        self.sentence_length = sentence_len

        if len(self.vocabulary_frequency_cutoff) != len(self.channeltypes):
            raise ValueError('Number of vocabulary frequency cutoffs have ' +
                             'to match number of channels')

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
        for freq, channeltype in zip(self.vocabulary_frequency_cutoff,
                                     self.channeltypes):
            self.channels.append(
                vocabulary_factory(channeltype,
                                   freq, linegen(), self.sentence_length))

    def generate_authors(self, linereader):

        for i, line in enumerate(
                linereader.readlines(skipfirst=True)):
            author, date, text = line.split(';')
            text = util.clean(text)

            if len(text) > 30000:
                print('WARNING: Skipping text longer than 30,000 characters ' +
                      'on line {}'.format(i + 1))
            elif len(text) < 400:
                print('WARNING: Skipping text shorter than 400 characters ' +
                      'on line {}'.format(i + 1))
            elif len(sent_tokenize(text)) > 500:
                print('WARNING: Skipping text with more than 500 sentences ' +
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
            raise Exception(
                'read_encoded_line does not currently \
                        work with global padding')

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
                known_inputs, unknown_inputs, y = self.generate_batch(
                    batch, reader)
                yield known_inputs + unknown_inputs, y

    def generate_batch(self, batch, linereader):
        knowns = []
        unknowns = []

        y = np.zeros((self.batch_size, abs(int(self.binary) - 2)))

        for (i, (known, unknown, label)) in enumerate(batch):
            knowns.append(self.read_encoded_line(linereader, known))
            unknowns.append(self.read_encoded_line(linereader, unknown))

            y[i] = self.label_true if label else self.label_false

        X_knowns = []
        X_unknowns = []

        for i, channel in enumerate(self.channels):
            known_channel = [x[i] for x in knowns]
            unknown_channel = [x[i] for x in unknowns]

            min_known = min_unknown = None
            pad = channel.padding

            if self.batch_normalization == 'truncate':
                truncs = [(len(x), len(y)) for x, y in
                          zip(known_channel, unknown_channel)]
                pad = self.pad
                min_known, min_unknown = (min(x) for x in zip(*truncs))

            X_known = sequence.pad_sequences(
                known_channel,
                value=pad,
                maxlen=min_known,
                truncating='post',
                padding='post'
            )

            X_unknown = sequence.pad_sequences(
                unknown_channel,
                value=pad,
                maxlen=min_unknown,
                truncating='post',
                padding='post'
            )

            X_knowns.append(X_known)
            X_unknowns.append(X_unknown)

        return X_knowns, X_unknowns, y
