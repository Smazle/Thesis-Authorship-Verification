# Authorship Verification
## Deep Learning Based Methods for Authorship Verification

Examples of how to run networks.

### python -m src.networks.generic\_network

#### usage: generic\_network.py 
[-h]  [--history HISTORY] [--graph GRAPH] [--weights WEIGHTS] [--epochs EPOCHS]
[--retry [RETRY]] networkname {load-reader,create-reader} ...

Simple NN for authorship verification

positional arguments:

  networkname           Which network to train.
  {load-reader,create-reader}
    load-reader         Load a reader from a file.
    create-reader       Create a new reader from arguments.

optional arguments:
  -h, --help            show this help message and exit
  --history HISTORY     Path to file to write history to.
  --graph GRAPH         Path to file to visualize network in.
  --weights WEIGHTS     Use the weights given as start weights instead of
                        randomly initializing.
  --epochs EPOCHS       How many epochs to run.
  --retry [RETRY]       Should the network keep trying using a reduced
                        batch_size?


#### usage: generic\_network.py networkname load-reader 
[-h] reader

positional arguments:
  reader      Use this pickled reader and not a new reader.

optional arguments:
  -h, --help  show this help message and exit




#### usage: generic\_network.py networkname create-reader 

[-h] [-b BATCH_SIZE] 
[-vfc VOCABULARY\_FREQUENCY\_CUTOFF [VOCABULARY\_FREQUENCY\_CUTOFF ...]]
[-bn BATCH\_NORMALIZATION] [--pad PAD] [--binary BINARY]
[--channels CHANNELS [CHANNELS ...]]
[-sl [SENTENCE\_LENGTH]]
[training\_file]
[validation\_file]

positional arguments:
  training\_file         Path to file containing training data.
  validation\_file       Path to file containing validation data.

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH\_SIZE, --batch-size BATCH\_SIZE
                        Size of batches.
  -vfc VOCABULARY\_FREQUENCY\_CUTOFF [VOCABULARY\_FREQUENCY\_CUTOFF ...], --vocabulary-frequency-cutoff VOCABULARY_FREQUENCY_CUTOFF [VOCABULARY_FREQUENCY_CUTOFF ...]
                        Characters with a frequency below this threshold is
                        ignored by thereader. Providing several applies a
                        differnet theshold to the differnetchannels
  -bn BATCH\_NORMALIZATION, --batch-normalization BATCH\_NORMALIZATION
                        Either "pad" or "truncate". Batches will be normalized
                        using thismethod.
  --pad PAD             Whether or not to pad all texts to length of longest
                        text.
  --binary BINARY       Whether to run reader with binary crossentropy or
                        categorical crossentropy
  --channels CHANNELS [CHANNELS ...]
                        Which channels to use.
  -sl [SENTENCE\_LENGTH], --sentence-length [SENTENCE\_LENGTH]
                        If channel SENTENCE is used. This determines the
                        length of each sentence



#### Network Names Available:
* CNN's
    * network2
    * network3
    * network4
    * network5
    * network6
* RNN's
    * r\_network1
    * r\_network2
    * r\_network3
    * r\_network4
    * r\_network5
    * r\_network6
    * r\_network7
    * r\_network8
    * r\_network9
    * r\_network10

