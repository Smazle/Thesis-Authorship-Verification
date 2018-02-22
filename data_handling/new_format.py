#!/usr/bin/python3

import numpy as np
import sys

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

    # List of lines each new author start on.
    authors = []

    def __init__(self, filepath, batch_size=32, newline='$NL$',
            semicolon='$SC$'):

        self.filepath = filepath
        self.f = open(self.filepath, 'r')

        self.generate_seek_positions()
        self.generate_author_start()

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

        print(self.authors)
        self.f.seek(self.line_offset[4])
        print(self.f.read(80))
        self.f.seek(self.line_offset[5])
        print(self.f.read(80))

    # Read in the file once and build a list of line offsets.
    def generate_seek_positions(self):
        offset = 0
        for line in self.f:
            self.line_offset.append(offset)
            offset += len(line.encode('utf-8'))
        self.f.seek(0)

    def generate_author_start(self):
        self.f.seek(self.line_offset[1])

        prev_author = None
        for i, line in enumerate(self.f):
            i = i + 1
            author, text = line.split(';')
            if author != prev_author:
                self.authors.append(i)

            prev_author = author
        self.f.seek(0)

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
