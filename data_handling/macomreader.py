#!/usr/bin/python3

import itertools
import numpy as np
import random
import sys
import time


# Class assume that authors are in order. It will not work if they are not in
# order.
class MacomReader:

    # The maximal length of any of the texts.
    max_len = 0

    # A set containing all the different characters used in the input file.
    vocabulary = set()

    # Mapping from a character to its one-hot encoding.
    vocabulary_map = {}

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

    # List of offsets the lines in the datafile start on.
    line_offset = []

    # Map from author identifier to list of line numbers.
    authors = {}

    # List of problems which consist of two line indices to two texts and either
    # 1 or 0. If 1 the texts are from the same author and if 0 they are from
    # different authors.
    problems = []

    # Which encoding to encode the characters in.
    encoding = None

    def __init__(self, filepath, batch_size=32, newline='$NL$',
                 semicolon='$SC$', encoding='one-hot'):

        # Save parameters.
        self.filepath = filepath
        self.batch_size = batch_size
        self.newline = newline
        self.semicolon = semicolon
        self.encoding = encoding

        self.f = open(self.filepath, 'r')

        # Generate representation used to generate training data.
        self.generate_seek_positions()
        self.generate_authors()
        self.generate_problems()
        self.generate_vocabulary_map()

    def generate_vocabulary_map(self):
        self.f.seek(self.line_offset[1])

        for line in self.f:
            author, text = line.split(';')
            decoded = unescape(text, self.newline, self.semicolon)

            self.vocabulary = self.vocabulary.union(decoded)

            if len(decoded) > self.max_len:
                self.max_len = len(decoded)

        if self.encoding == 'one-hot':
            encoding = np.diag(np.ones(len(self.vocabulary) + 1))
        elif self.encoding == 'numbers':
            encoding = list(range(0, len(self.vocabulary) + 1))

        for i, c in enumerate(self.vocabulary):
            self.vocabulary_map[c] = encoding[i]

        self.padding = encoding[-1]

    # Read in the file once and build a list of line offsets.
    def generate_seek_positions(self):
        self.f.seek(0)

        offset = 0
        for line in self.f:
            self.line_offset.append(offset)
            offset += len(line.encode('utf-8'))
        self.f.seek(0)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

    # Generate batches of samples.
    def generate(self):
        problems = itertools.cycle(self.problems)

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

    def read_line(self, line):
        self.f.seek(self.line_offset[line])
        author, text = self.f.readline().split(';')
        unescaped = unescape(text, self.newline, self.semicolon)
        encoded = list(map(lambda x: self.vocabulary_map[x], unescaped))

        len_diff = self.max_len - len(encoded)
        padded = encoded + ([self.padding] * len_diff)

        return np.array(padded)


# Replace escapes in the string from the MaCom dataset.
def unescape(text, newline, semicolon):
    return text.replace(newline, '\n').replace(semicolon, ';')


if __name__ == '__main__':
    with MacomReader(sys.argv[1], 64, encoding='numbers') as generator:
        t = time.time()
        for batch in generator.generate():
            print(time.time() - t)
            t = time.time()
