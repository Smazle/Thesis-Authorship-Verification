# -*- coding: utf-8 -*-
# !/usr/bin/python3

from enum import Enum
from collections import Counter
import numpy as np
from ..util import utilities as util


class ChannelType(Enum):
    CHAR = 'char'
    WORD = 'word'


class Vocabulary:

    def __init__(self, vocabulary_frequency_cutoff, str_generator):

        if vocabulary_frequency_cutoff < 0 or vocabulary_frequency_cutoff > 1:
            raise Exception('vocabulary_frequency_cutoff should be between 0 and 1')

        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff
        self.vocabulary = set()
        self.vocabulary_usage = Counter()
        self.max_len = 0

        for txt in str_generator:
            self.add_vocabulary(txt)

        total = sum(self.vocabulary_usage.values())
        self.vocabulary_frequencies = {k: v / total for k, v in
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
            self.vocabulary_map[c] = encoding[i + 2]

        self.garbage = encoding[1]

    def add_vocabulary(self, txt):
        sequence = self.split_to_sequence(txt)

        self.vocabulary = self.vocabulary.union(sequence)
        self.vocabulary_usage = self.vocabulary_usage + Counter(sequence)

        if len(sequence) > self.max_len:
            self.max_len = len(sequence)

    def encode(self, txt):
        sequence = self.split_to_sequence(txt)
        return np.array(list(map(lambda x: self.vocabulary_map[x]
                        if x in self.vocabulary_map else
                        self.garbage, sequence)))

    def split_to_sequence(self, txt):
        raise NotImplementedError('Subclasses should implement this.')


class CharVocabulary(Vocabulary):

    def split_to_sequence(self, txt):
        return txt


class WordVocabulary(Vocabulary):

    def split_to_sequence(self, txt):
        return util.wordProcess(txt)


def vocabulary_factory(channeltype, vocabulary_frequency_cutoff, strgen):
    if channeltype == ChannelType.CHAR:
        return CharVocabulary(vocabulary_frequency_cutoff, strgen)
    elif channeltype == ChannelType.WORD:
        return WordVocabulary(vocabulary_frequency_cutoff, strgen)
    else:
        raise Exception('Illegal state, unknown channel.')
