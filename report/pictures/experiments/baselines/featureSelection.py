# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np

f1 = sys.argv[-2]
f2 = sys.argv[-1]

data1 = np.loadtxt(f1, dtype=float, delimiter=',')
data2 = np.loadtxt(f2, dtype=float, delimiter=',')

m = min(len(data1), len(data2))
data1 = data1[:m]
data2 = data2[:m]

X = range(len(data1))

print([int(x[0]) for x in sorted(data1, key=lambda x:x[-1])[:50]])

m = [np.argmax(data1[:, -1]), np.argmax(data2[:, -1])]
y = [data1[:, -1][m[0]], data2[:, -1][m[1]]]

print m, y

accuracy1, = plt.plot(X, data1[:, -1], label='SVM Accuracy', c='blue')
accuracy2, = plt.plot(
    X, data2[:, -1], label='Extended Delta Accuracy', c='green')
m = plt.scatter(m, y, label='Peak', c='red')

plt.title('Feature Selection Accuracy')
plt.xlabel('Number of features')
plt.ylabel('Validation Accuracy')
plt.legend(handles=[accuracy1, accuracy2, m])
plt.grid(True)
plt.savefig('FeatureSelect.png')
# plt.show()
