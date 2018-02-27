#!/usr/bin/python3

import itertools
import numpy as np
import random
import sys
from collections import Counter, defaultdict


# Class assume that authors are in order. It will not work if they are not in
# order.
class MacomReader:

    # The maximal length of any of the texts.
    max_len = 0

    # Number between 0 and 1 (inclusive) cutoff point for when a character is
    # replaced with the default character.
    vocabulary_frequency_cutoff = 0.0

    # A set containing all the different characters used in the input file.
    vocabulary = set()

    # Mapping from a character to its one-hot encoding.
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

    # The one hot encoding we use to represent padding of texts.
    padding = None

    # The path of the datafile.
    filepath = None

    # The size of each batch we yield in the generator.
    batch_size = None

    # What string to replace to a newline.
    newline = None

    # What string to replace with a semicolon.
    semicolon = None

    # Open datafile.
    f = None

    # Open datafile in binary mode.
    fb = None

    # List of offsets the lines in the datafile start on.
    line_offset = []

    # Map from author identifier to list of line numbers.
    authors = {}

    # List of problems which consist of two line indices to two texts and
    # either 1 or 0. If 1 the texts are from the same author and if 0 they are
    # from different authors.
    problems = []

    # Which encoding to encode the characters in.
    encoding = None

    # If 0..8 80% is training data and 20% are validation data.
    validation_split = None

    # List of training problems.
    training_problems = None

    # List of validation problems.
    validation_problems = None

    # TODO: Take argument specifying whether or not to ignore first line in
    # file.
    def __init__(self, filepath, batch_size=32, newline='$NL$',
                 semicolon='$SC$', encoding='one-hot', validation_split=0.8,
                 vocabulary_frequency_cutoff=0.0):

        if encoding != 'one-hot' and encoding != 'numbers':
            raise ValueError('encoding should be "one-hot" or "numbers"')

        if validation_split > 1.0 or validation_split < 0.0:
            raise ValueError('validation_split between 0 and 1 required')

        if vocabulary_frequency_cutoff > 1.0 or\
                vocabulary_frequency_cutoff < 0.0:
            raise ValueError('vocabulary_frequency_cutoff between 0 and 1 ' +
                             'required')

        # Save parameters.
        self.filepath = filepath
        self.batch_size = batch_size
        self.newline = newline
        self.semicolon = semicolon
        self.encoding = encoding
        self.validation_split = validation_split
        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff

    def generate_training(self):
        return self.generate(self.training_problems)

    def generate_validation(self):
        return self.generate(self.validation_problems)

    def __enter__(self):
        self.f = open(self.filepath, 'r')
        self.fb = open(self.filepath, 'rb')

        # Generate representation used to generate training data.
        self.generate_seek_positions()
        self.generate_authors()
        self.generate_problems()
        self.generate_vocabulary_map()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        self.fb.close()

    def generate_vocabulary_map(self):
        self.f.seek(self.line_offset[1])

        for line in self.f:
            author, text = line.split(';')
            decoded = unescape(text, self.newline, self.semicolon)

            self.vocabulary = self.vocabulary.union(decoded)
            self.vocabulary_usage = self.vocabulary_usage + Counter(decoded)

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

        if self.encoding == 'one-hot':
            encoding = np.diag(np.ones(len(self.vocabulary_above_cutoff) + 2))
        elif self.encoding == 'numbers':
            encoding = list(range(0, len(self.vocabulary_above_cutoff) + 2))

        self.vocabulary_map = defaultdict(lambda: encoding[-2])

        for i, c in enumerate(self.vocabulary_above_cutoff):
            self.vocabulary_map[c] = encoding[i]

        self.padding = encoding[-1]

    # Read in the file once and build a list of line offsets.
    def generate_seek_positions(self):
        self.fb.seek(0)

        offset = 0
        for line in self.fb:
            self.line_offset.append(offset)
            offset += len(line)
        self.fb.seek(0)

    def generate_authors(self):
        self.f.seek(self.line_offset[1])

        for i, line in enumerate(self.f):
            author, text = line.split(';')
            try:
                self.authors[author].append(i + 1)
            except KeyError:
                self.authors[author] = [i + 1]

    # TODO: Make sure the same file is not returned for the same author.
    def generate_problems(self):

        for author in self.authors:
            same1 = random.choice(self.authors[author])
            same2 = random.choice(self.authors[author])

            all_authors = set(self.authors.keys())
            all_authors.remove(author)

            different = random.choice(self.authors[random.choice(list(all_authors))])

            self.problems.append((same1, same2, 1))
            self.problems.append((same1, different, 0))

        random.shuffle(self.problems)

        split_point = int(len(self.problems) * self.validation_split)
        self.training_problems = self.problems[:split_point]
        self.validation_problems = self.problems[split_point:]

    def read_line(self, line):
        self.f.seek(self.line_offset[line])
        author, text = self.f.readline().split(';')
        unescaped = unescape(text, self.newline, self.semicolon)
        encoded = list(map(lambda x: self.vocabulary_map[x], unescaped))

        len_diff = self.max_len - len(encoded)
        padded = encoded + ([self.padding] * len_diff)

        return np.array(padded)

    # Generate batches of samples.
    def generate(self, problems):
        problems = itertools.cycle(problems)

        while True:
            batch = itertools.islice(problems, self.batch_size)

            if self.encoding == 'one-hot':
                X_known = np.zeros((self.batch_size, self.max_len,
                                   len(self.vocabulary) + 1))
                X_unknown = np.zeros((self.batch_size, self.max_len,
                                     len(self.vocabulary) + 1))
                y = np.zeros((self.batch_size, 2))
            elif self.encoding == 'numbers':
                X_known = np.zeros((self.batch_size, self.max_len))
                X_unknown = np.zeros((self.batch_size, self.max_len))
                y = np.zeros((self.batch_size, 2))

            for (i, (line1, line2, label)) in enumerate(batch):
                (text1, text2) = (self.read_line(line1), self.read_line(line2))
                X_known[i] = text1
                X_unknown[i] = text2

                if label == 0:
                    y[i] = np.array([1, 0])
                else:
                    y[i] = np.array([0, 1])

            yield [X_known, X_unknown], y


# Replace escapes in the string from the MaCom dataset.
def unescape(text, newline, semicolon):
    return text.replace(newline, '\n').replace(semicolon, ';')


if __name__ == '__main__':
    reader = MacomReader(
        sys.argv[1],
        vocabulary_frequency_cutoff=1 / 100000,
        encoding='numbers'
    )

    with reader as generator:
        print(len(generator.vocabulary_above_cutoff))
        print(generator.vocabulary_above_cutoff)
        print(len(generator.vocabulary_below_cutoff))
        print(generator.vocabulary_below_cutoff)
        print(generator.read_line(10))
