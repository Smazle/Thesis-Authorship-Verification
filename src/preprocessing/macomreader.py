#!/usr/bin/python3

import itertools
import numpy as np
import random
import sys
from collections import Counter, defaultdict
import pickle


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

    # The encoding we use to represent padding of texts.
    padding = None

    # The encoding we use to represent garbage characters.
    garbage = None

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

    # If not None the reader will save the state of the reader to the filename.
    save_file = None

    # TODO: Take argument specifying whether or not to ignore first line in
    # file.
    def __init__(self, filepath, batch_size=32, newline='$NL$',
                 semicolon='$SC$', encoding='one-hot', validation_split=0.8,
                 vocabulary_frequency_cutoff=0.0, save_file=None):

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
        self.save_file = save_file

        self.f = open(self.filepath, mode='r', encoding='utf-8')
        self.fb = open(self.filepath, mode='rb')
        self.f_val = open(self.filepath, mode='r', encoding='utf-8')

        # Generate representation used to generate training data.
        self.generate_seek_positions()
        self.generate_authors()
        self.generate_problems()
        self.generate_vocabulary_map()

        # Close files.
        self.f.close()
        self.fb.close()
        self.f_val.close()

    def generate_training(self):
        return self.generate(self.training_problems, self.f)

    def generate_validation(self):
        return self.generate(self.validation_problems, self.f_val)

    def __enter__(self):
        self.f = open(self.filepath, mode='r', encoding='utf-8')
        self.f_val = open(self.filepath, mode='r', encoding='utf-8')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        self.fb.close()
        self.f_val.close()

        if self.save_file is not None:
            print('I AM RUNNING')
            with open(self.save_file, 'wb') as save_here:
                pickle.dump(self, save_here)

    def generate_vocabulary_map(self):
        for author in self.authors:
            for line in self.authors[author]:
                self.f.seek(self.line_offset[line])
                text = self.f.readline()
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

        self.vocabulary_map = {}

        for i, c in enumerate(self.vocabulary_above_cutoff):
            self.vocabulary_map[c] = encoding[i]

        self.padding = encoding[-1]
        self.garbage = encoding[-2]


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

            if len(text) > 30000:
                print('WARNING: Skipping text longer than 30,000 characters ' +
                      'on line {}'.format(i + 1))
            elif len(text) < 200:
                print('WARNING: Skipping text shorter than 200 characters ' +
                      'on line {}'.format(i + 1))
            else:
                try:
                    self.authors[author].append(i + 1)
                except KeyError:
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

    def read_line(self, line, f):
        f.seek(self.line_offset[line])
        author, text = f.readline().split(';')
        unescaped = unescape(text, self.newline, self.semicolon)

        encoded = list(map(lambda x: self.vocabulary_map[x]
            if x in self.vocabulary_map else self.garbage, unescaped))

        len_diff = self.max_len - len(encoded)
        padded = encoded + ([self.padding] * len_diff)

        return np.array(padded)

    # Generate batches of samples.
    def generate(self, problems, f):
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
                X_known[i] = self.read_line(line1, f)
                X_unknown[i] = self.read_line(line2, f)

                if label == 0:
                    y[i] = np.array([1, 0])
                else:
                    y[i] = np.array([0, 1])

            yield [X_known, X_unknown], y

    def save_reader(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    # Declare which properties should be saved.
    def __getstate__(self):
        return (self.max_len, self.vocabulary_frequency_cutoff, self.vocabulary,
                self.vocabulary_map, self.vocabulary_usage,
                self.vocabulary_frequencies, self.vocabulary_above_cutoff,
                self.vocabulary_below_cutoff, self.padding, self.garbage,
                self.filepath, self.batch_size, self.newline, self.semicolon,
                self.line_offset, self.authors, self.problems, self.encoding,
                self.validation_split, self.training_problems,
                self.validation_problems, self.save_file)

    # Declare how to read properties from a pickled object.
    def __setstate__(self, state):
        (self.max_len, self.vocabulary_frequency_cutoff, self.vocabulary,
            self.vocabulary_map, self.vocabulary_usage,
            self.vocabulary_frequencies, self.vocabulary_above_cutoff,
            self.vocabulary_below_cutoff, self.padding, self.garbage,
            self.filepath, self.batch_size, self.newline, self.semicolon,
            self.line_offset, self.authors, self.problems, self.encoding,
            self.validation_split, self.training_problems,
            self.validation_problems, self.save_file) = state


def load_reader(filename):
    return pickle.load(open(filename, 'rb'))


# Replace escapes in the string from the MaCom dataset.
def unescape(text, newline, semicolon):
    return text.replace(newline, '\n').replace(semicolon, ';')


if __name__ == '__main__':
    reader = MacomReader(
        sys.argv[1],
        vocabulary_frequency_cutoff=1 / 100000,
        encoding='numbers',
        validation_split=0.95
    )

    with reader as generator:
        print(len(reader.training_problems))
        print(len(reader.validation_problems))
