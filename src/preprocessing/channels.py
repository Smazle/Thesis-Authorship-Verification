# -*- coding: utf-8 -*-
# !/usr/bin/python3


class ChannelType(Enum):
    CHAR = 'char'
    WORD = 'word'


class CharVocabulary:

    def __init__(self, vocabulary_frequency_cutoff, str_generator):

        if vocabulary_frequency_cutoff < 0 or vocabulary_frequency_cutoff > 1:
            raise Exception('vocabulary_frequency_cutoff should be between 0 and 1')

        self.vocabulary_frequency_cutoff = vocabulary_frequency_cutoff
        self.vocabulary = set()
        self.vocabulary_usage = Counter()
        self.max_len = 0

        for txt in str_generator:
            self.add_vocabulary(txt)

        total_chars = sum(self.vocabulary_usage.values())
        self.vocabulary_frequencies = {k: v / total_chars for k, v in
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
        self.vocabulary = self.vocabulary.union(txt)
        self.vocabulary_usage = self.vocabulary_usage + Counter(txt)

        if len(txt) > self.max_len:
            self.max_len = len(decoded)

    def encode(self, chars):
        return np.array(map(lambda x: self.vocabulary_map[x]
                        if x in self.vocabulary_map else
                        self.garbage, chars))
