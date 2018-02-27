import numpy as np
import random
from feature_extractor import analyze_input_folder  # , FeatureExtractor
from itertools import combinations


random.seed = 7


dataFolder = '../../../data/pan/'
outfile = 'output'

authors = analyze_input_folder(dataFolder)


N = []
N.extend(range(2, 11))
N.extend(range(15, 25, 5))
# N.extend(range(20, 40, 10))
# N.extend(range(40, 120, 20))


S = []
S.extend(range(10, 40, 10))
S.extend(range(40, 100, 20))
S.extend(range(100, 550, 50))

print(N)
print(S)

POS = 219
SPEC = 524
CHAR = 2178
WORD = 188472
FREQ = 27535


calls = [POS, SPEC, CHAR, WORD, FREQ]

POS = SPEC = CHAR = WORD = FREQ = []

for i in S:
    for q in range(1, len(N)):
        grams = combinations(N, q)

        for combo in grams:
            combo = list(combo)
            postag_grams = list(map(lambda x: (x, i), combo))
            special_grams = list(map(lambda x: (x, i), combo))
            char_grams = list(map(lambda x: (x, i), combo))
            word_grams = list(map(lambda x: (x, i), combo))
            word_frequencies = i

            POS.append(postag_grams)
            SPEC.append(special_grams)
            CHAR.append(char_grams)
            WORD.append(word_grams)
            FREQ.append(word_frequencies)
print(np.array(POS).shape)

#    feature_extractor = FeatureExtractor(authors,
#            postag_grams=postag_grams,
#            special_character_grams=postag_grams,
#            word_grams=postag_grams,
#            word_frequencies=1,
#            character_grams=postag_grams)
#
#    feature_extractor.fit(outfile + "_" + str(i))
