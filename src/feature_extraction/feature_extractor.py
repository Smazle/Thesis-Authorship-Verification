# -*- coding: utf-8 -*-

import os
from .character import CharacterNGramFeatureExtractor,\
    SpecialCharacterNGramFeatureExtractor
from .posTag import PosTagNGramsExtractor
from .words import WordFrequencyExtractor, WordNGramsFeatureExtractor
import numpy as np
from nltk.corpus import europarl_raw
import pickle
from ..util.utilities import clean
import time

# TODO: description.


class FeatureExtractor:

    def __init__(self, authors, character_grams=[], special_character_grams=[],
                 word_frequencies=0, postag_grams=[], word_grams=[]):

        self.authors = authors

        # Generate corpus from nltk danish text. In that way we have no bias
        # towards the training data.
        self.corpus = gen_corpus()

        # Create feature extractors for the types of features requested.
        self.extractors = []
        self.feature_names = []

        # Handle character n-grams.
        for (n, size) in character_grams:
            extractor = CharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['char-' + str(n)] * size

            print('Char-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle special character n-grams.
        for (n, size) in special_character_grams:
            extractor = SpecialCharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['spec-' + str(n)] * size

            print('Special-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word frequencies.
        if word_frequencies != 0:
            extractor = WordFrequencyExtractor(word_frequencies)
            extractor.fit(self.corpus)
            self.feature_names += ['freq'] * word_frequencies

            print('Word Frequencies fitted, %d of total %d' %
                  (word_frequencies, extractor.max))

            self.extractors.append(extractor)

        # Handle POS tagging n-grams.
        for (n, size) in postag_grams:
            extractor = PosTagNGramsExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['pos-' + str(n)] * size

            print('POS-Tag-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word n-grams.
        for (n, size) in word_grams:
            extractor = WordNGramsFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['word-' + str(n)] * size

            print('Word-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

    def extract(self, outfile):
        with open(outfile, 'a') as f:
            # Write header.
            f.write('author' + ','.join(self.feature_names) + '\r\n')

            print('Starting to generate features')
            for i, [author, date, text] in enumerate(self.authors):
                start = time.time()
                text = clean(text)

                try:
                    features = self.extract_features(text)
                    t = time.time() - start
                    print('Text', i, '-', t * 1000)
                except ZeroDivisionError:
                    print('Text', i, 'Err')
                    f.write('Err' + '\n')
                    continue

                line = [int(author) + features]
                f.write(','.join(list(map(str, features))) + '\r\n')

    def extract_features(self, text):
        features = []

        for extractor in self.extractors:
            features = features + extractor.extract(text)

        return features


def gen_corpus():
    chapters = europarl_raw.danish.chapters()
    vals = ['%', ',', ':', ')', '(']

    txt = ''

    for chapter in chapters:

        for sentence in chapter:
            start = True

            if len(sentence) == 1:
                txt += ' ' + sentence[0]
            else:
                start = True
                skip = False
                for i, word in enumerate(sentence[:-1]):

                    if skip:
                        skip = False
                        continue

                    if word in vals:
                        continue

                    if sentence[i - 1] == '(':
                        txt += ' ' + '(' + word
                        continue

                    if word == "\"":
                        if start:
                            txt += ' ' + "\"" + sentence[i + 1]
                            skip = True
                        else:
                            txt += "\""
                        start = not start
                        # import pdb; pdb.set_trace()
                        continue

                    if i + 1 < len(sentence) and sentence[i + 1] in vals:
                        txt += ' ' + word + sentence[i + 1]
                        if sentence[i + 1] == "\"":
                            start = not start

                        continue

                    txt += ' ' + word

                txt += sentence[-1]

        txt += '\n'

    return txt
