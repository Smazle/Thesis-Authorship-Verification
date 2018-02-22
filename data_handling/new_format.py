#!/usr/bin/python3

import numpy as np
import sys
import random

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

    def __init__(self, filepath, batch_size=32, newline='$NL$',
            semicolon='$SC$'):

        self.filepath = filepath
        self.f = open(self.filepath, 'r')

        self.generate_seek_positions()
        self.generate_authors()
        self.generate_problems()

        for line in self.f:
            author, text = line.split(';')
            decoded = unescape(text, newline, semicolon)

            if len(decoded) > self.max_len:
                self.max_len = len(decoded)
                self.vocabulary = self.vocabulary.union(set(decoded))

        one_hot = np.diag(np.ones(len(self.vocabulary) + 1))

        for i, c in enumerate(self.vocabulary):
            self.vocabulary_map[c] = one_hot[i]

        padding = one_hot[-1]

        print(self.problems)

    # Read in the file once and build a list of line offsets.
    def generate_seek_positions(self):
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
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

    # Generate batches of samples.
    def generate(self):
        pass

# Replace escapes in the string from the MaCom dataset.
def unescape(text, newline, semicolon):
    return text.replace(newline, '\n').replace(semicolon, ';')

if __name__ == '__main__':
    with MacomReader(sys.argv[1]) as generator:
        pass
