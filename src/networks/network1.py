#!/usr/bin/python3

from keras.layers import Dense, Convolution1D, GlobalMaxPooling1D, Input,\
    Concatenate, Embedding
from keras.models import Model
from ..preprocessing import MacomReader
import argparse

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument('datafile', type=str, help='Path to data file.')
args = parser.parse_args()

reader = MacomReader(
    args.datafile,
    batch_size=32,
    encoding='numbers'
)

with reader as generator:
    inshape = (generator.max_len, )

    known_in = Input(shape=inshape)
    unknown_in = Input(shape=inshape)

    embedding = Embedding(len(generator.vocabulary) + 1, 5,
                          input_length=generator.max_len)

    conv = Convolution1D(filters=1000, kernel_size=10, strides=1,
                         activation='relu')

    repr_known = GlobalMaxPooling1D()(conv(embedding(known_in)))
    repr_unknown = GlobalMaxPooling1D()(conv(embedding(unknown_in)))

    full_input = Concatenate()([repr_known, repr_unknown])

    dense = Dense(500, activation='relu')(full_input)
    dense = Dense(500, activation='relu')(full_input)
    dense = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[known_in, unknown_in], outputs=dense)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    steps_n = len(generator.problems) / generator.batch_size
    model.fit_generator(
        generator=generator.generate(),
        steps_per_epoch=steps_n,
        epochs=100
    )
