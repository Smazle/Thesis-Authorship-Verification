#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
from keras import backend as K
import argparse
import jsonpickle
from ..preprocessing import LineReader
from ..util import utilities as util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Parse arguments.
parser = argparse.ArgumentParser(
    description='Output what the filters react to in each text.'
)
parser.add_argument(
    'reader',
    type=str,
    help='Path to reader that can read and encode the lines.'
)
parser.add_argument(
    'model',
    type=str,
    help='Path to model that can predict.'
)
args = parser.parse_args()

model = load_model(args.model)

get_output = K.function([
    model.get_layer('known_input').input,
    model.get_layer('unknown_input').input,
    K.learning_phase()], [
    model.get_layer('embedding_1').get_output_at(0)]
)

with open(args.reader, 'r') as macomreader_in:
    reader = jsonpickle.decode(macomreader_in.read())

original_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZæÆøØåÅ'
characters = list(filter(lambda x: x in reader.vocabulary_map, original_characters))
missing = set(original_characters) - set(characters)
print('Missing', missing)
encoded = np.array(list(map(lambda x: reader.vocabulary_map[x], characters)))
encoded = encoded.reshape((1, encoded.shape[0]))
layer_output = get_output([encoded, encoded, 0])[0][0]

pca = PCA(n_components=2)
X = pca.fit_transform(layer_output)

for letter, encoded in zip(characters, X):
    if letter.islower():
        size = 40
    else:
        size = 80

    if letter.lower() in 'aeiouyæøå':
        color = 'b'
    else:
        color = 'g'

    plt.scatter(encoded[0], encoded[1], marker=r"$ {} $".format(letter), s=size, c=color)

plt.grid(True)
plt.show()
