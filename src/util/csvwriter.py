
import math
import numpy as np
from keras.callbacks import Callback


class CSVWriter(Callback):

    def __init__(self, validation_generator, validation_steps,
                 training_generator, training_steps, outfile,
                 continue_train=False):

        self.continue_train = continue_train

        self.validation_generator = validation_generator
        self.validation_steps = validation_steps

        self.training_generator = training_generator
        self.training_steps = training_steps

        if self.continue_train:
            with open(outfile) as f:
                self.prev_epochs = sum(1 for line in f) - 1

            # Open in append mode to not overwrite previous values.
            self.outfile = open(outfile, 'a')
        else:
            self.outfile = open(outfile, 'w')
            self.prev_epochs = 1

            self.outfile.write(
                'epoch,' +
                'accuracy,' +
                'validation_accuracy,' +
                'true_positives,' +
                'true_negatives,' +
                'false_positives,' +
                'false_negatives\r\n')

            self.outfile.flush()

    def on_train_begin(self, logs={}):
        if not self.continue_train:
            val_acc, val_tps, val_tns, val_fps, val_fns = self.compute_metrics(
                self.validation_generator,
                self.validation_steps
            )

            acc, _, _, _, _ = self.compute_metrics(
                self.training_generator,
                self.training_steps
            )

            self.outfile.write('{},{},{},{},{},{},{}\r\n'.format(
                0, acc, val_acc, val_tps, val_tns, val_fps, val_fns))

            self.outfile.flush()

        return

    def on_train_end(self, logs={}):
        self.outfile.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        _, tps, tns, fps, fns = self.compute_metrics(
            self.validation_generator,
            self.validation_steps
        )

        self.outfile.write('{},{},{},{},{},{},{}\r\n'.format(
            epoch + self.prev_epochs, logs['acc'], logs['val_acc'], tps, tns,
            fps, fns))
        self.outfile.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def compute_metrics(self, generator, steps):
        tps = 0
        tns = 0
        fps = 0
        fns = 0

        for i in range(math.ceil(steps)):
            X, y = next(generator)
            prediction = self.model.predict(X)

            prediction = prediction[:, 1] - prediction[:, 0]
            prediction = prediction > 0

            y = y[:, 1] - y[:, 0]
            y = y > 0

            tps += np.sum(np.logical_and(prediction == y, y))
            tns += np.sum(np.logical_and(prediction == y, np.logical_not(y)))
            fps += np.sum(np.logical_and(prediction != y, y))
            fns += np.sum(np.logical_and(prediction != y, np.logical_not(y)))

        accuracy = (tps + tns) / (tps + tns + fps + fns)

        return accuracy, tps, tns, fps, fns
