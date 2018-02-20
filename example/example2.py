#!/usr/bin/python3

from keras.layers import Dense, Flatten, Merge, Convolution1D,\
        GlobalMaxPooling1D, Input, Embedding, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

conv = Convolution1D(filters=100, kernel_size=10, strides=1, activation='relu')

known_in = Input(shape=(100,))  # TODO: Change to length of longest text.
unknown_in = Input(shape=(100,))

known = Embedding(128, 10, init='random_uniform')(known_in)
unknown = Embedding(128, 10, init='random_uniform')(unknown_in)

repr_known = GlobalMaxPooling1D()(conv(known))
repr_unknown = GlobalMaxPooling1D()(conv(unknown))

full_input = Concatenate()([repr_known, repr_unknown])

dense = Dense(100, activation='relu')(full_input)
dense = Dense(2, activation='relu')(dense)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Model(inputs=[known_in, unknown_in], outputs=dense)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
