
import numpy as np
import math
from keras.callbacks import CSVLogger, Callback

class CSVWriter(Callback):

    def __init__(self, validation_generator, validation_steps, outfile):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.outfile = open(outfile, 'w')
        self.outfile.write('accuracy,validation_accuracy,true_positives,' +
            'true_negatives,false_positives,false_negatives\r\n')
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

        self.outfile.write('{},{},{},{},{},{}\r\n'.format(
            logs['acc'], logs['val_acc'], tps, tns, fps, fns))
        self.outfile.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
