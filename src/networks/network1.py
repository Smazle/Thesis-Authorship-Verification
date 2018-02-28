#!/usr/bin/python3

from keras.layers import Dense, Convolution1D, GlobalMaxPooling1D, Input,\
    Concatenate, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from ..preprocessing import MacomReader
import argparse
import math
import numpy as np
import resource


gb4 = 4000000000  # 4 GB in bytes.
gb6 = 6000000000  # 6 GB in bytes.

# Limit memory usage of the script so we don't crash a computer.
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (gb6, hard))

# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument('datafile', type=str, help='Path to data file.')
args = parser.parse_args()


class MyCallback(Callback):

    def __init__(self, validation_generator, validation_steps, outfile):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.outfile = open(outfile, 'w')
        self.outfile.write('tps,tns,fps,fns\r\n')
        self.outfile.flush()

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        self.outfile.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        tps = 0
        tns = 0
        fps = 0
        fns = 0

        for i in range(math.ceil(self.validation_steps)):
            X, y = next(self.validation_generator)
            prediction = self.model.predict(X)

            prediction = prediction[:, 1] - prediction[:, 0]
            prediction = prediction > 0

            y = y[:, 1] - y[:, 0]
            y = y > 0

            tps += np.sum(np.logical_and(prediction == y, y))
            tns += np.sum(np.logical_and(prediction == y, np.logical_not(y)))
            fps += np.sum(np.logical_and(prediction != y, y))
            fns += np.sum(np.logical_and(prediction != y, np.logical_not(y)))

        self.outfile.write('{},{},{},{}\r\n'.format(tps, tns, fps, fns))
        self.outfile.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


reader = MacomReader(
    args.datafile,
    batch_size=8,
    encoding='numbers',
    vocabulary_frequency_cutoff=1 / 100000
)

with reader as generator:
    inshape = (generator.max_len, )

    known_in = Input(shape=inshape)
    unknown_in = Input(shape=inshape)

    embedding = Embedding(len(generator.vocabulary_above_cutoff) + 2, 5,
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

    steps_n = len(generator.training_problems) / generator.batch_size
    val_steps_n = len(generator.validation_problems) / generator.batch_size

    callbacks = [
        ModelCheckpoint(
            'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True
        ),
        CSVLogger('history.csv'),
        MyCallback(generator.generate_validation(), val_steps_n,
                   'history2.csv')
    ]

    model.fit_generator(
        generator=generator.generate_training(),
        steps_per_epoch=steps_n,
        epochs=100,
        validation_data=generator.generate_validation(),
        validation_steps=val_steps_n,
        callbacks=callbacks
    )
