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
import pandas as pd
from nltk.tokenize import sent_tokenize


class FeatureExtractor:

    def __init__(self, authors, character_grams=[], special_character_grams=[],
                 word_frequencies=0, postag_grams=[],
                 word_grams=[], corpus=None):

        self.authors = authors

        # Generate corpus from nltk danish text. In that way we have no bias
        # towards the training data.
        if corpus is not None:
            self.corpus = gen_own_corpus(corpus)
        else:
            self.corpus = gen_corpus()

        # Create feature extractors for the types of features requested.
        self.extractors = []
        self.feature_names = []
        self.actual_features = []

        print('Fitting Corpus')

        # Handle character n-grams.
        for (n, size) in character_grams:
            extractor = CharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['char-' + str(n)] * size
            for i in range(size):
                self.actual_features.append(
                    'char-{}-{}\t{}'.format(n, i, repr(extractor.grams[i])))

            print('... Char-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle special character n-grams.
        for (n, size) in special_character_grams:
            extractor = SpecialCharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['spec-' + str(n)] * size
            for i in range(size):
                self.actual_features.append(
                    'special-{}-{}\t{}'.format(n, i, repr(extractor.grams[i])))

            print('... Special-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word frequencies.
        if word_frequencies != 0:
            extractor = WordFrequencyExtractor(word_frequencies)
            extractor.fit(self.corpus)
            self.feature_names += ['freq'] * word_frequencies
            for i in range(word_frequencies):
                self.actual_features.append(
                    'word-{}\t{}'.format(i, repr(extractor.words[i])))

            print('... Word Frequencies fitted, %d of total %d' %
                  (word_frequencies, extractor.max))

            self.extractors.append(extractor)

        # Handle POS tagging n-grams.
        for (n, size) in postag_grams:
            extractor = PosTagNGramsExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['pos-' + str(n)] * size
            for i in range(size):
                self.actual_features.append(
                    'pos-{}-{}\t{}'.format(n, i, repr(extractor.grams[i])))

            print('... POS-Tag-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word n-grams.
        for (n, size) in word_grams:
            extractor = WordNGramsFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.feature_names += ['word-' + str(n)] * size
            for i in range(size):
                self.actual_features.append(
                    'word-{}-{}\t{}'.format(n, i, repr(extractor.grams[i])))

            print('... Word-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # open('Features.Names', 'w').write('\n'.join(self.actual_features))

    def extract(self, outfile):
        with open(outfile, 'a') as f:
            # Write header.
            f.write('author' + ','.join(self.feature_names) + '\r\n')

            for i, [author, date, text] in enumerate(self.authors):
                start = time.time()
                text = clean(text)

                try:
                    features = self.extract_features(text)
                    t = time.time() - start
                    print('Text', i, '-', t * 1000)
                except Exception as e:
                    print('Text', i, 'Err', str(e))
                    continue

                line = [author] + features
                f.write(','.join(list(map(str, line))) + '\r\n')

    def extract_features(self, text):
        features = []

        for extractor in self.extractors:
            features = features + extractor.extract(text)

        return features


def gen_own_corpus(corpus):
    print('Generating Corpus')
    print('... Loading File')
    dataFrame = pd.read_csv(corpus, sep=';')
    data = dataFrame.as_matrix(columns=['Text']).flatten()

    print('... Applying Filters')
    data = list(map(lambda x: clean(x), data))

    print('... Joining')
    data = '\n'.join(data)

    print('... Done')
    return data


def gen_corpus():
    chapters = europarl_raw.danish.chapters()
    cleanUpVals = ['%', ',', ':', ')', '(']

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

                    if word in cleanUpVals:
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
                        continue

                    if i + 1 < len(sentence) and sentence[i + 1] \
                            in cleanUpVals:
                        txt += ' ' + word + sentence[i + 1]
                        if sentence[i + 1] == "\"":
                            start = not start

                        continue

                    txt += ' ' + word

                txt += sentence[-1]

        txt += '\n'

    print(txt)
    return txt
