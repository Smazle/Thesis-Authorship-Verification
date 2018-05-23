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
import tensorflow as tf
import json

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
    help='Path to file to visualize network in.',
    default=None
)
parser.add_argument(
    '--weights',
    type=str,
    help='Use the weights given as start weights instead of randomly' +
    ' initializing.',
    default=None
)
parser.add_argument(
    '--epochs',
    type=int,
    help='How many epochs to run.',
    # default=100
    default=None
)
parser.add_argument(
    '--retry',
    type=bool,
    help='Should the network keep trying using a reduced batch_size?',
    default=None,
    nargs='?'
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
    help='Use this pickled reader and not a new reader.',
    default=None
)
# Create reader part.
create_reader = readerparser.add_parser(
    'create-reader',
    help='Create a new reader from arguments.'
)
create_reader.add_argument(
    'training_file',
    type=str,
    help='Path to file containing training data.',
    nargs='?',
    default=None
)
create_reader.add_argument(
    'validation_file',
    type=str,
    help='Path to file containing validation data.',
    nargs='?',
    default=None
)
create_reader.add_argument(
    '-b',
    '--batch-size',
    type=int,
    help='Size of batches.',
    default=None
)
create_reader.add_argument(
    '-vfc',
    '--vocabulary-frequency-cutoff',
    type=float,
    help='Characters with a frequency below this threshold is ignored by the' +
    'reader. Providing several applies a differnet theshold to the differnet' +
    'channels',
    default=None,
    nargs='+'
)
create_reader.add_argument(
    '-bn',
    '--batch-normalization',
    type=str,
    help='Either "pad" or "truncate". Batches will be normalized using this' +
    'method.',
    default=None
)
create_reader.add_argument(
    '--pad',
    help='Whether or not to pad all texts to length of longest text.',
    default=None,
    type=bool
)
create_reader.add_argument(
    '--binary',
    help='Whether to run reader with binary crossentropy or categorical ' +
    'crossentropy',
    type=bool,
    default=None
)
create_reader.add_argument(
    '--channels',
    help='Which channels to use.',
    nargs='+',
    default=None
)
create_reader.add_argument(
    '-sl',
    '--sentence-length',
    type=int,
    help='If channel SENTENCE is used.\
          This determines the length of each sentence',
    default=None,
    nargs='?'
)

args = parser.parse_args()

# TODO:Find smart way to do pathing
config = json.load(
    open('./src/networks/config/{}_config.json'.format(args.networkname), 'r'))

var_args = vars(args)
for key in var_args.keys():
    if var_args[key] is None:
        var_args[key] = config[key]

# Either load reader from file or create a new one.
if hasattr(args, 'reader') and args.reader is not None:
    print('Loading reader from {}'.format(args.reader))
    with open(args.reader, mode='r') as reader_in:
        reader = jsonpickle.decode(reader_in.read())
else:
    channels = list(map(lambda x: ChannelType(x), args.channels))

    print(('Creating new MaCom reader with parameters, batch_size={}, ' +
           'vocabulary_frequency_cutoff={}, batch_normalization={}, ' +
           'pad={}, binary={}, channels={}, sentence_length={}'
           ).format(args.batch_size,
                    args.vocabulary_frequency_cutoff,
                    args.batch_normalization,
                    args.pad, args.binary,
                    channels,
                    args.sentence_length
                    ))

    reader = MacomReader(
        args.training_file,
        args.validation_file,
        batch_size=args.batch_size,
        vocabulary_frequency_cutoff=args.vocabulary_frequency_cutoff,
        batch_normalization=args.batch_normalization,
        pad=args.pad,
        binary=args.binary,
        channels=channels,
        sentence_len=args.sentence_length
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

while True:
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
    except tf.errors.ResourceExhaustedError:
        reader.batch_size = int(reader.batch_size / 2)
        if args.retry and reader.batch_size >= 1:

            steps_n = len(reader.training_problems) / reader.batch_size
            val_steps_n = len(reader.validation_problems) / reader.batch_size

            args.weights = 'final_model.hdf5'
            print('MEMORY ERROR, RUNNING AGAIN WITH BATCH SIZE {}'
                  .format(reader.batch_size))
            continue
    finally:
        model.save('final_model.hdf5')

    break
