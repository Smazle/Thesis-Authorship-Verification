#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import tempfile
import unittest
from src.preprocessing.macomreader import *
from src.preprocessing.channels import *
import src.util.utilities as util


class TestMacomReader(unittest.TestCase):

    def test_read_encoded_line_1(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=1, pad=False,
                batch_normalization='pad', min_len_characters=0
            )

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 1)
                self.assertEqual(len(encoded_line[0]),
                                 len('This is the first text\n'))

    # Reading the same line twice should yield the same result.
    def test_read_encoded_line_2(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=1, pad=False,
                batch_normalization='pad', min_len_characters=0
            )

            with LineReader(f.name) as linereader:
                encoded_line1 = reader.read_encoded_line(linereader, 2)
                encoded_line2 = reader.read_encoded_line(linereader, 2)

                self.assertTrue(np.allclose(encoded_line1, encoded_line2))

    # Reading with $NL$ should convert that to '\n'.
    def test_read_encoded_line_3(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=1, pad=False,
                batch_normalization='pad', min_len_characters=0
            )

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 4)

                self.assertEqual(
                    encoded_line[0][5], reader.channels[0].vocabulary_map['\n']
                )

    # Reading with $NL$ should convert that to '\n'.
    def test_read_encoded_line_4(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=1, pad=False,
                batch_normalization='pad', min_len_characters=0,
                channels=[ChannelType.SENTENCE],
                vocabulary_frequency_cutoff=[0.0],
                sentence_len=10
            )

            mapping = reader.channels[0].word_vocab.vocabulary_map

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 5)[0]
                self.assertEqual(encoded_line.shape, (3, 10))

                line1 = np.array([mapping['multiple']] + ([0] * 9))
                line2 = np.array([mapping['sentences']] + ([0] * 9))
                line3 = np.array([mapping['test']] + ([0] * 9))

                self.assertTrue((encoded_line[0] == line1).all())
                self.assertTrue((encoded_line[1] == line2).all())
                self.assertTrue((encoded_line[2] == line3).all())

    def test_generate_batch_1(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=2, pad=False,
                batch_normalization='pad', min_len_characters=0
            )

            problems = [(1, 2, 1), (2, 3, 0)]

            with LineReader(f.name) as linereader:
                knowns, unknowns, labels = reader.generate_batch(
                    problems, linereader)
                self.assertEqual(len(knowns), 1)  # We use one channel.
                self.assertEqual(len(unknowns), 1)  # We use one channel.
                self.assertEqual(len(labels), 2)  # We have a batch size of 2.
                self.assertTrue((labels[1] == np.array([1, 0])).all())
                self.assertTrue((labels[0] == np.array([0, 1])).all())

                known_len = max(f.text_lengths[1], f.text_lengths[2])
                self.assertEqual(knowns[0].shape, (2, known_len))

                unknown_len = max(f.text_lengths[2], f.text_lengths[3])
                self.assertEqual(unknowns[0].shape, (2, unknown_len))

    def test_generate_batch_2(self):
        with FileOne() as f:
            reader = MacomReader(
                f.name, f.name, batch_size=2, pad=False,
                batch_normalization='pad', min_len_characters=0,
                channels=[ChannelType.CHAR, ChannelType.WORD],
                vocabulary_frequency_cutoff=[0.0, 0.0]
            )

            problems = [(1, 2, 1), (2, 3, 0)]

            with LineReader(f.name) as linereader:
                knowns, unknowns, labels = reader.generate_batch(
                    problems, linereader)
                self.assertEqual(len(knowns), 2)  # We use two channels.
                self.assertEqual(len(unknowns), 2)  # We use two channels.
                self.assertEqual(len(labels), 2)  # We have a batch size of 2.
                self.assertTrue((labels[1] == np.array([1, 0])).all())
                self.assertTrue((labels[0] == np.array([0, 1])).all())


class FileOne:

    lines = [
        b'ID;Date;Text\n',
        b'author1;11-04-18;This is the first text\n',
        b'author1;11-04-17;This is the second text\n',
        b'author2;12-05-44;This is a text from another mother\n',
        b'author2;11-11-94;This $NL$ text $NL$ contains $NAME$.\n',
        b'author3;11-12-88;Multiple. Sentences. Test.\n'
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
