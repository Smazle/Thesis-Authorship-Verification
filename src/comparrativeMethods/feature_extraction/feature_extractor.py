import os
from character import CharacterNGramFeatureExtractor,\
    SpecialCharacterNGramFeatureExtractor
from posTag import PosTagNGramsExtractor
from words import WordFrequencyExtractor, WordNGramsFeatureExtractor
import numpy as np
from sklearn.preprocessing import scale
from nltk.corpus import europarl_raw


# TODO: description.
class FeatureExtractor:

    def __init__(
        self, authors, character_grams=[], special_character_grams=[],
            word_frequencies=0, postag_grams=[], word_grams=[],
            normalize=True, feature_header=None):

        self.authors = authors
        self.normalize = normalize
        self.feature_header = feature_header

        # If the type is Normal only unique files are used. If it is pancho
        # all files are concatenated.
        self.corpus = gen_corpus()

        # Create feature extractors for the types of features requested.
        self.extractors = []
        self.featureNames = []

        # Handle character n-grams.
        for (n, size) in character_grams:
            extractor = CharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.featureNames += extractor.chosen_features()

            print('Char-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle special character n-grams.
        for (n, size) in special_character_grams:
            extractor = SpecialCharacterNGramFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.featureNames += extractor.chosen_features()

            print('Special-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word frequencies.
        if word_frequencies != 0:
            extractor = WordFrequencyExtractor(word_frequencies)
            extractor.fit(self.corpus)
            self.featureNames += extractor.chosen_features()

            print('Word Frequencies fitted, %d of total %d' %
                  (word_frequencies, extractor.max))

            self.extractors.append(extractor)

        # Handle POS tagging n-grams.
        for (n, size) in postag_grams:
            extractor = PosTagNGramsExtractor(n, size)
            extractor.fit(self.corpus)
            self.featureNames += extractor.chosen_features()

            print('POS-Tag-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

        # Handle word n-grams.
        for (n, size) in word_grams:
            extractor = WordNGramsFeatureExtractor(n, size)
            extractor.fit(self.corpus)
            self.featureNames += extractor.chosen_features()

            print('Word-%d-grams fitted, %d of total %d' %
                  (n, size, extractor.max))

            self.extractors.append(extractor)

    def extract(self, outfile, master_file=None):
        # Generate features for each author.
        author_features = []
        for author in self.authors:

            for known in author.texts:
                known_features = self.extract_features(known)

                features = known_features + [author.id]
                author_features.append(features)

        # Write features to file.
        author_features = np.array(author_features)
        if self.normalize:
            author_features[:, :-1] = scale(author_features[:, :-1], axis=0)

        np.savetxt(outfile, author_features)

        if self.feature_header is not None:
            open(self.feature_header, 'w').write('ø'.join(self.featureNames))

    def extract_features(self, text):
        features = []

        for extractor in self.extractors:
            features = features + extractor.extract(text)

        return features


# TODO: description.
# class Author:
#
#    def __init__(self, name, known_files, unknown_file, truth):
#        self.name = name
#        self.truth = truth
#
#        known = map(lambda x: open(x, 'r').read(), known_files)
#        self.known = list(known)
#
#        self.unknown = open(unknown_file, 'r').read()
#
#    def __str__(self):
#        return 'Author: ' + self.name


class Author:
    def __init__(self, authorID, texts):
        self.id = authorID
        self.texts = [texts.replace('$NL$', '\n').replace('$SC$', ';')]

    def __str__(self):
        return 'Author: ' + self.id


# TODO: description.
# Return [Author]
# def analyse_input_folder(data_folder):
#    all_fnames = sorted(os.listdir(data_folder))
#    author_folders = filter(lambda x: x.startswith('EN'), all_fnames)
#    truth_f = open(data_folder + '/truth.txt', 'r', encoding='utf-8-sig')\
#        .read()
#
#    authors = []
#    for folder in author_folders:
#        all_fnames = os.listdir(data_folder + '/' + folder)
#        all_fnames = sorted(all_fnames)
#
#        known = filter(lambda x: x.startswith('known'), all_fnames)
#        known = map(lambda x: data_folder + '/' + folder + '/' + x, known)
#
#        unknown = data_folder + '/' + folder + '/unknown.txt'
#
#        truth = filter(lambda x: x.startswith(folder), truth_f.split('\n'))
#        truth = list(truth)[0].endswith('Y')
#
#        name = int(folder[2:])
#
#        authors.append(Author(name, list(known), unknown, truth))
#
#    return authors


def analyze_input_folder(data_folder):
    files = (x for x in os.listdir(data_folder) if '.csv' in x)

    texts = []

    for f in files:
        authors = [Author(int(x), y) for x, y
                   in np.loadtxt(data_folder + '/' + f,
                                 delimiter=';', skiprows=1, dtype=str)]
        texts = np.concatenate((texts, authors))

    return texts


def check(inp, vals):
    return inp in vals


def gen_corpus():
    if 'corpus' not in os.listdir('.'):
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

                        if sentence[i-1] == '(':
                            txt += ' ' + '(' + word
                            continue

                        if word == "\"":
                            if start:
                                txt += ' ' + "\"" + sentence[i+1]
                                skip = True
                            else:
                                txt += "\""
                            start = not start
                            # import pdb; pdb.set_trace()
                            continue

                        if i+1 < len(sentence) and sentence[i+1] in vals:
                            txt += ' ' + word + sentence[i+1]
                            if sentence[i+1] == "\"":
                                start = not start

                            continue

                        txt += ' ' + word

                    txt += sentence[-1]

            txt += '\n'

        open('corpus', 'w').write(txt)
        return txt
    else:
        return open('corpus', 'r').read()
