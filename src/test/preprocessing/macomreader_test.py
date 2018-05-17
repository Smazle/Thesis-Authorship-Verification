#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import tempfile
import unittest
from src.preprocessing.macomreader import *

class TestMacomReader(unittest.TestCase):

    def test_read_encoded_line(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'ID;Date;Text\n')
            f.write(b'author1;11-04-18;This is the first text\n')
            f.write(b'author1;11-04-17;This is the second text\n')
            f.flush()

            reader = MacomReader(
                f.name, batch_size=1, pad=False, batch_normalization='pad',
                min_len_characters=0, ignore_n_characters=0)

            with LineReader(f.name) as linereader:
                encoded_line = reader.read_encoded_line(linereader, 1)
                self.assertEqual(len(encoded_line), len('This is the first text'))

if __name__ == '__main__':
    unittest.main()
