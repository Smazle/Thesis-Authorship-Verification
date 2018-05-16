#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .network_factory import construct_network, Network
import argparse
from ..util import CSVWriter
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from ..preprocessing import MacomReader
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from ..preprocessing.channels import ChannelType


# Make sure that jsonpickle works on numpy arrays.
jsonpickle_numpy.register_handlers()


# Parse arguments.
parser = argparse.ArgumentParser(
    description='Simple NN for authorship verification'
)
parser.add_argument(
    'networkname',
    type=str,
    help='Which network to train.'
)
parser.add_argument(
    '--history',
    type=str,
    help='Path to file to write history to.',
    default=None
)
parser.add_argument(
    '--graph',
    type=str,
    help='Path to file to visualize network in.'
)
parser.add_argument(
    '--weights',
    type=str,
    help='Use the weights given as start weights instead of randomly' +
         ' initializing.'
)
parser.add_argument(
    '--epochs',
    type=int,
    help='How many epochs to run.',
    default=100
)

# Parse either a filepath to a reader to load or arguments to create a new
# reader.
readerparser = parser.add_subparsers()
# Load reader part.
load_reader = readerparser.add_parser(
    'load-reader',
    help='Load a reader from a file.'
)
load_reader.add_argument(
    'reader',
    type=str,
    help='Use this pickled reader and not a new reader.'
)
# Create reader part.
create_reader = readerparser.add_parser(
    'create-reader',
    help='Create a new reader from arguments.'
)
create_reader.add_argument(
    'datafile',
    type=str,
    help='Path to data file.'
)
create_reader.add_argument(
    '--validation-split',
    type=float,
    help='How much data to use as the validation set vs the training set.',
    default=0.95
)
create_reader.add_argument(
    '--batch-size',
    type=int,
    help='Size of batches.',
    default=8
)
create_reader.add_argument(
    '--vocabulary-frequency-cutoff',
    type=float,
    help='Characters with a frequency below this threshold is ignored by the' +
         'reader',
    default=1 / 100000
)
create_reader.add_argument(
    '--batch-normalization',
    type=str,
    help='Either "pad" or "truncate". Batches will be normalized using this' +
         'method.',
    default='pad'
)
create_reader.add_argument(
    '--pad',
    dest='pad',
    help='Whether or not to pad all texts to length of longest text.',
    default=False,
    action='store_true'
)
create_reader.add_argument(
    '--binary',
    dest='binary',
    help='Whether to run reader with binary crossentropy or categorical ' +
         'crossentropy',
    default=False,
    action='store_true'
)
create_reader.add_argument(
    '--word',
    dest='word',
    help='Whether to use characters or words as input to the networks. Both ' +
         'will be changed to a sequence of ints.',
    default=False,
    action='store_true'
)
args = parser.parse_args()

# Either load reader from file or create a new one.
if hasattr(args, 'reader') and args.reader is not None:
    print('Loading reader from {}'.format(args.reader))
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())
else:
    print(('Creating new MaCom reader with parameters, batch_size={}, ' +
           'vocabulary_frequency_cutoff={}, batch_normalization={}, ' +
           'pad={}, binary={}, char={}'
           ).format(args.batch_size,
                    args.vocabulary_frequency_cutoff,
                    args.batch_normalization,
                    args.pad, args.binary,
                    not args.word))

    reader = MacomReader(
        args.datafile,
        batch_size=args.batch_size,
        vocabulary_frequency_cutoff=args.vocabulary_frequency_cutoff,
        validation_split=args.validation_split,
        batch_normalization=args.batch_normalization,
        pad=args.pad,
        binary=args.binary,
        channels= [ChannelType.WORD] if args.word else [ChannelType.CHAR]
    )

    print('Writing new MaCom reader to reader.p')
    with open('reader.p', mode='w') as reader_out:
        reader_out.write(jsonpickle.encode(reader))

steps_n = len(reader.training_problems) / reader.batch_size
val_steps_n = len(reader.validation_problems) / reader.batch_size

model = construct_network(Network(args.networkname), reader)

# Setup callbacks.
callbacks = [
    ModelCheckpoint(
        'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True
    )
]

if args.history is not None:
    callbacks.append(
        CSVWriter(
            reader.generate_validation(), val_steps_n,
            reader.generate_training(), steps_n, args.history,
            args.weights is not None
        )
    )

# If we are asked to visualize model, do so.
if args.graph is not None:
    plot_model(model, to_file=args.graph, show_shapes=True)

# If we are given weights, load them.
if args.weights is not None:
    model.load_weights(args.weights)

try:
    model.fit_generator(
        generator=reader.generate_training(),
        steps_per_epoch=steps_n,
        epochs=args.epochs,
        validation_data=reader.generate_validation(),
        validation_steps=val_steps_n,
        callbacks=callbacks
    )
finally:
    # TODO: Load best weights.
    model.save('final_model.hdf5')
