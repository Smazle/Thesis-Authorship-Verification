# -*- coding: utf-8 -*-
# !/usr/bin/python3

import itertools
import numpy as np
import random
from collections import Counter
import pickle
import sys
from ..util import utilities as util
from datetime import datetime


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

    # A set containing all the different characters used in the input file.
    vocabulary = set()

    # Mapping from a character to its encoding.
    vocabulary_map = {}

    # Mapping from a characters to the number of times it is used in the
    # dataset given.
    vocabulary_usage = Counter()

    # Mapping from a character to its frequency.
    vocabulary_frequencies = {}

    # Set of characters that are above the cutoff given.
    vocabulary_above_cutoff = {}

    # Set of characters that are below the cutoff given (they are ignored).
    vocabulary_below_cutoff = {}

    # The encoding we use to represent padding of texts.
    padding = None

    # The encoding we use to represent garbage characters.
    garbage = None

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

    # Whether or not to word encode or character.
    char = True

    # TODO: Take argument specifying whether or not to ignore first line in
    # file.
    def __init__(self, filepath, batch_size=32, char=True,
                 validation_split=0.8, vocabulary_frequency_cutoff=0.0,
                 pad=True, binary=False):

        if validation_split > 1.0 or validation_split < 0.0:
            raise ValueError('validation_split between 0 and 1 required')

        if vocabulary_frequency_cutoff > 1.0 or\
                vocabulary_frequency_cutoff < 0.0:
            raise ValueError('vocabulary_frequency_cutoff between 0 and 1 ' +
                             'required')

        # Save parameters.
        self.char = char
        self.filepath = filepath
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff
        self.binary = binary
        self.pad = pad

        # Generate representation used to generate training data.
        with LineReader(self.filepath) as linereader:
            self.generate_authors(linereader)
            self.generate_problems()
            self.generate_vocabulary_map(linereader)

    def generate_training(self):
        return self.generate(self.training_problems) if \
            self.pad else self.padless_generate(self.training_problems)

    def generate_validation(self):
        return self.generate(self.validation_problems) if \
            self.pad else self.padless_generate(self.validation_problems)

    def generate_vocabulary_map(self, linereader):
        for author in self.authors:
            for line_n in self.authors[author]:
                autor, date, text = linereader.readline(line_n).split(';')
                decoded = util.clean(text)
                if not self.char:
                    decoded = util.wordProcess(decoded)

                self.vocabulary = self.vocabulary.union(decoded)
                self.vocabulary_usage = self.vocabulary_usage + \
                    Counter(decoded)

                if len(decoded) > self.max_len:
                    self.max_len = len(decoded)

        total_chars = sum(self.vocabulary_usage.values())
        self.vocabulary_frequencies = {k: v / total_chars for k, v in
                                       self.vocabulary_usage.items()}

        self.vocabulary_above_cutoff =\
            {k for k, v in self.vocabulary_frequencies.items()
             if v > self.vocabulary_frequency_cutoff}
        self.vocabulary_below_cutoff =\
            {k for k, v in self.vocabulary_frequencies.items()
             if v < self.vocabulary_frequency_cutoff}

        encoding = list(range(0, len(self.vocabulary_above_cutoff) + 2))

        self.vocabulary_map = {}

        for i, c in enumerate(self.vocabulary_above_cutoff):
            self.vocabulary_map[c] = encoding[i]

        self.padding = encoding[-1]
        self.garbage = encoding[-2]

    def generate_authors(self, linereader):
        for i, line in enumerate(linereader.readlines(skipfirst=True)):
            author, date, text = line.split(';')

            if len(text) > 30000:
                print('WARNING: Skipping text longer than 30,000 characters ' +
                      'on line {}'.format(i + 1))
            elif len(text) < 200:
                print('WARNING: Skipping text shorter than 200 characters ' +
                      'on line {}'.format(i + 1))
            elif author in self.authors:
                self.authors[author].append(i + 1)
            else:
                self.authors[author] = [i + 1]

    # TODO: The split into train and validation should happen between authors
    # and not just specific training instances.
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

    def read_encoded_line(self, linereader, line_n, with_date=False):
        author, date, text = linereader.readline(line_n).split(';')
        unescaped = util.clean(text)

        if not self.char:
            unescaped = util.wordProcess(text)

        encoded = list(map(lambda x: self.vocabulary_map[x]
                           if x in self.vocabulary_map else
                           self.garbage, unescaped))

        if self.pad:
            len_diff = self.max_len - len(encoded)
            padded = encoded + ([self.padding] * len_diff)
        else:
            padded = encoded

        if with_date:
            epoch = datetime.utcfromtimestamp(0)
            date = datetime.strptime(date, '%d-%m-%Y')
            time = (date - epoch).total_seconds()
            return np.array(padded), time
        else:
            return np.array(padded)

    # Generate batches of samples.
    def generate(self, problems):
        with LineReader(self.filepath, encoding='utf-8') as reader:
            problems = itertools.cycle(problems)

            while True:
                batch = itertools.islice(problems, self.batch_size)

                X_known = np.zeros((self.batch_size, self.max_len))
                X_unknown = np.zeros((self.batch_size, self.max_len))
                y = np.zeros((self.batch_size, 2))

                for (i, (line1, line2, label)) in enumerate(batch):
                    X_known[i] = self.read_encoded_line(reader, line1)
                    X_unknown[i] = self.read_encoded_line(reader, line2)

                    if not self.binary:
                        y = np.array([1, 0]) if y == 0 else np.array([0, 1])

                yield [X_known, X_unknown], y

    def padless_generate(self, problems):
        with LineReader(self.filepath, encoding='utf-8') as reader:
            problems = itertools.cycle(problems)

            while True:
                known_line, unknown_line, y = next(problems)

                known = self.read_encoded_line(reader, known_line)
                unknown = self.read_encoded_line(reader, unknown_line)

                known = known.reshape(1, len(known))
                unknown = unknown.reshape(1, len(unknown))

                if not self.binary:
                    y = np.array([1, 0]) if y == 0 else np.array([0, 1])
                    y = y.reshape(1, 2)
                else:
                    y = [y]

                yield [known, unknown], y


if __name__ == '__main__':
    reader1 = MacomReader(
        sys.argv[1],
        vocabulary_frequency_cutoff=1 / 100000,
        validation_split=0.95,
        char=True,
        batch_size=1,
        pad=False
    )

    print(reader1.authors)

    reader2 = pickle.load(open(sys.argv[2], mode='rb'))
    print(reader2.authors)
