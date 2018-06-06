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
    training_authors = {}
    validaiton_authors = {}

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

    def __init__(self, training_file, validation_file, batch_size=32,
                 vocabulary_frequency_cutoff=[0.0], pad=True, binary=False,
                 batch_normalization='truncate', channels=[ChannelType.CHAR],
                 sentence_len=None, max_len_characters=30000,
                 max_len_sentences=500, min_len_characters=200):

        if batch_normalization not in ['truncate', 'pad']:
            raise ValueError('Only truncate and pad is currently supported.')

        for channel in channels:
            if channel not in [ChannelType.CHAR, ChannelType.WORD,
                               ChannelType.SENTENCE]:
                raise ValueError(
                    'Only char, word or sentence channels allowed')

        if len(vocabulary_frequency_cutoff) != len(channels):
            raise ValueError('Number of vocabulary frequency cutoffs have ' +
                             'to match number of channels')

        # Save parameters
        self.training_file = training_file
        self.validation_file = validation_file
        self.batch_size = batch_size
        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff
        self.binary = binary
        self.pad = pad
        self.batch_normalization = batch_normalization
        self.channeltypes = channels
        self.sentence_length = sentence_len
        self.max_len_characters = max_len_characters
        self.max_len_sentences = max_len_sentences
        self.min_len_characters = min_len_characters

        if self.binary:
            self.label_true = np.array([1])
            self.label_false = np.array([0])
        else:
            self.label_true = np.array([0, 1]).reshape(1, 2)
            self.label_false = np.array([1, 0]).reshape(1, 2)

        # Generate representation used to generate training data.
        with LineReader(self.training_file) as linereader:
            self.training_authors = self.generate_authors(linereader)
            self.training_problems = self.generate_problems(self.training_authors)
            self.generate_vocabulary_maps(linereader, self.training_authors)

        # Generate validation problems.
        with LineReader(self.validation_file) as linereader:
            self.validation_authors = self.generate_authors(linereader)
            self.validation_problems = self.generate_problems(self.validation_authors)

    def generate_training(self):
        return self.generate(self.training_file, self.training_problems)

    def generate_validation(self):
        return self.generate(self.validation_file, self.validation_problems)

    def generate_authors(self, linereader):

        authors = {}

        for i, line in enumerate(linereader.readlines(skipfirst=True)):
            author, date, text = line.split(';')
            text = util.clean(text)

            assert len(text) <= self.max_len_characters
            assert len(text) >= self.min_len_characters
            assert len(sent_tokenize(text)) <= self.max_len_sentences

            if author in authors:
                authors[author].append(i + 1)
            else:
                authors[author] = [i + 1]

        return authors

    def generate_problems(self, authors):

        problems = []

        for author in authors:
            other = set(authors.keys())
            other.remove(author)

            # Generate all combinations of the authors texts.
            for (l1, l2) in itertools.combinations(authors[author], r=2):
                # Generate a sample with same author.
                problems.append((l1, l2, 1))

                # Generate a sample with different authors.
                same = random.choice(authors[author])
                different = random.choice(authors[random.choice(list(other))])
                problems.append((same, different, 0))

        random.shuffle(problems)

        return problems

    def generate_vocabulary_maps(self, linereader, authors):
        def linegen():
            for author in authors:
                for line_n in authors[author]:
                    autor, date, text = linereader.readline(line_n).split(';')
                    decoded = util.clean(text)
                    yield decoded

        self.channels = []
        for freq, channeltype in zip(self.vocabulary_frequency_cutoff,
                                     self.channeltypes):
            self.channels.append(
                vocabulary_factory(channeltype,
                                   freq, linegen(), self.sentence_length))

    # Returns a list of numpy arrays containing integers where each array is an
    # encoded sequence. The list ordering corresponds to the self.channels
    # parameter.
    def read_encoded_line(self, linereader, line_n, with_date=False):
        assert line_n > 0
        if self.pad:
            raise Exception(
                'read_encoded_line does not currently work with global padding'
            )

        author, date, text = linereader.readline(line_n).split(';')

        unescaped = util.clean(text)

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
    def generate(self, filename, problems):
        with LineReader(filename, encoding='utf-8') as reader:
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
