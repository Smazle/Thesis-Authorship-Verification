#!/usr/bin/python3

from keras.layers import Dense, Convolution1D,\
    GlobalMaxPooling1D, Input, Embedding, Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model
import numpy as np

X = np.load('../data_handling/X.npy')
y = np.load('../data_handling/y.npy')

split = int(len(X)/2)
print(split)
X_Known = X[:, :split]
X_Uknown = X[:, split:]

LENGTH = len(X[0])

conv = Convolution1D(filters=100, kernel_size=10, strides=1, activation='relu')

known_in = Input(shape=(LENGTH,))  # TODO: Change to length of longest text.
unknown_in = Input(shape=(LENGTH,))

known = Embedding(128, 10, init='random_uniform')(known_in)
unknown = Embedding(128, 10, init='random_uniform')(unknown_in)

repr_known = GlobalMaxPooling1D()(conv(known_in))
repr_unknown = GlobalMaxPooling1D()(conv(unknown_in))

full_input = Concatenate()([repr_known, repr_unknown])

dense = Dense(100, activation='relu')(full_input)
dense = Dense(2, activation='relu')(dense)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Model(inputs=[known_in, unknown_in], outputs=dense)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y = to_categorical(y)
print(X_Known.shape)
print(X_Uknown.shape)
model.fit([X_Known, X_Uknown], y)
