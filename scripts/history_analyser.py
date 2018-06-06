#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Generate graphs from history CSV output from neural ' +
    'network training')
parser.add_argument('datafile', type=str, help='Path to CSV file')
args = parser.parse_args()

with open(args.datafile, 'r') as csv_file:
    data = pd.read_csv(csv_file)

epochs = data.as_matrix(columns=['epoch']).flatten()
accuracies = data.as_matrix(columns=['accuracy']).flatten()
val_accuracies = data.as_matrix(columns=['validation_accuracy']).flatten()
tps = data.as_matrix(columns=['true_positives']).flatten()
tns = data.as_matrix(columns=['true_negatives']).flatten()
fps = data.as_matrix(columns=['false_positives']).flatten()
fns = data.as_matrix(columns=['false_negatives']).flatten()

print('Best validation', np.max(val_accuracies), 'in epoch',
      np.argmax(val_accuracies) + 1)

plt.plot(epochs, accuracies, c='r', label='Training')
plt.plot(epochs, val_accuracies, c='b', label='Validation')
plt.grid(True)
plt.ylim(0.0, 1.01)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
