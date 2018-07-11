# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np

f1 = sys.argv[1]
f2 = sys.argv[2]
cap = int(sys.argv[3])

data1 = np.loadtxt(f1, dtype=float, delimiter=',')
data2 = np.loadtxt(f2, dtype=float, delimiter=',')

data1 = data1[:cap]
data2 = data2[:cap]

m = min(len(data1), len(data2))
data1 = data1[:m]
data2 = data2[:m]

X = range(len(data1))

m = [np.argmax(data1[:, -1]), np.argmax(data2[:, -1])]
y = [data1[:, -1][m[0]], data2[:, -1][m[1]]]

print(m)
print(y)

accuracy1, = plt.plot(X, data1[:, -1], label='SVM Accuracy', c='blue')
accuracy2, = plt.plot(
    X, data2[:, -1], label='Extended Delta Accuracy', c='green')
m = plt.scatter(m, y, label='Peak', c='red')

plt.xlabel('Number of features')
plt.ylabel('Validation Accuracy')
plt.legend(handles=[accuracy1, accuracy2, m])
plt.grid(True)
plt.savefig('FeatureSelect.pdf', format='pdf')
# plt.show()
