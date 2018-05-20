#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import tempfile
import unittest
from src.preprocessing.macomreader import *
import src.util.utilities as util


class TestMacomReader(unittest.TestCase):

    def test_read_encoded_line_1(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, batch_size=1, pad=False, batch_normalization='pad',
                min_len_characters=0, ignore_n_characters=0)

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 1)
                self.assertEqual(len(encoded_line[0]),
                                 len('This is the first text\n'))

    # Reading the same line twice should yield the same result.
    def test_read_encoded_line_2(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, batch_size=1, pad=False, batch_normalization='pad',
                min_len_characters=0, ignore_n_characters=0)

            with LineReader(f.name) as linereader:
                encoded_line1 = reader.read_encoded_line(linereader, 2)
                encoded_line2 = reader.read_encoded_line(linereader, 2)

                self.assertTrue(np.allclose(encoded_line1, encoded_line2))

    # Reading with $NL$ should convert that to '\n'.
    def test_read_encoded_line_3(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, batch_size=1, pad=False, batch_normalization='pad',
                min_len_characters=0, ignore_n_characters=0
            )

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 4)

                self.assertEqual(
                    encoded_line[0][5], reader.channels[0].vocabulary_map['\n']
                )

    def test_generate_batch(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, batch_size=2, pad=False, batch_normalization='pad',
                min_len_characters=0, ignore_n_characters=0
            )

            problems = [(1, 2, 1), (2, 3, 0)]

            with LineReader(f.name) as linereader:
                knowns, unknowns, labels = reader.generate_batch(
                    problems, linereader)
                self.assertEqual(len(knowns), 1)  # We use one channel.
                self.assertEqual(len(unknowns), 1)  # We use one channel.
                self.assertEqual(len(labels), 2)  # We have a batch size of 2.

                problem_1_len = max(f.text_lengths[1], f.text_lengths[2])
                self.assertEqual(knowns[0].shape, (2, problem_1_len))


class FileOne:

    lines = [
        b'ID;Date;Text\n',
        b'author1;11-04-18;This is the first text\n',
        b'author1;11-04-17;This is the second text\n',
        b'author2;12-05-44;This is a text from another mother\n',
        b'author2;11-11-94;This $NL$ text $NL$ contains $NAME$.\n',
    ]

    text_lengths = list(
        map(lambda x: len(util.clean(x.decode('utf-8').split(';')[2])), lines))

    def __enter__(self):
        self.f = tempfile.NamedTemporaryFile().__enter__()
        for line in self.lines:
            self.f.write(line)
        self.f.flush()

        self.name = self.f.name

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()


if __name__ == '__main__':
    unittest.main()
