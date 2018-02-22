#!/usr/bin/python3

import numpy as np
import sys

class MacomReader:

    max_len = 0  # The maximal length of any of the texts.
    vocabulary = set()
    vocabulary_map = {}
    padding = None
    filepath = None
    problems = []

    line_seek_positions = []

    def __init__(self, filepath, batch_size=32, newline='$NL$',
            semicolon='$SC$'):

        self.filepath = filepath

        self.generate_seek_positions()

        with open(filepath) as f:
            for line in f:
                author, text = line.split(';')
                decoded = unescape(text, newline, semicolon)

                if len(decoded) > self.max_len:
                    self.max_len = len(decoded)
                    self.vocabulary = self.vocabulary.union(set(decoded))

        one_hot = np.diag(np.ones(len(self.vocabulary) + 1))

        for i, c in enumerate(self.vocabulary):
            self.vocabulary_map[c] = one_hot[i]

        padding = one_hot[-1]

    # Read in the file once and build a list of line offsets.
    def generate_seek_positions(self):
        line_offset = []
        offset = 0
        for line in file:
            line_offset.append(offset)
            offset += len(line)
        file.seek(0)

    def close():

    def __enter__(self):

    def __exit__(self, exc_type, exc_value, traceback)

# Now, to skip to line n (with the first line being line 0), just do
file.seek(line_offset[n])

    # Generate batches of samples.
    def generate(self):
        pass

# Replace escapes in the string from the MaCom dataset.
def unescape(text, newline, semicolon):
    return text.replace(newline, '\n').replace(semicolon, ';')

if __name__ == '__main__':
    reader = MacomReader(sys.argv[1])
