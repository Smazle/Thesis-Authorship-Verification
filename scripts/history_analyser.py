#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description=('Generate graphs from history CSV output from neural ' +
        'network training'
    )
)
parser.add_argument(
    'datafile',
    type=str,
    help='Path to CSV file'
)
args = parser.parse_args()

epochs_col = 0
accuracies_col = 1
validation_accuracies_col = 2
true_positives_col = 3
true_negatives_col = 4
false_positives_col = 5
false_negatives_col = 6

data = np.loadtxt(args.datafile, dtype=np.float, delimiter=',', skiprows=1)

plt.plot(data[:, epochs_col], data[:, accuracies_col], c='r', label='Training')
plt.plot(data[:, epochs_col], data[:, validation_accuracies_col], c='b',
         label='Validation')
plt.grid(True)
plt.ylim(0.0, 1.01)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
