#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Return uniform distribution over the timestamps.
def uniform(xs):
    return np.ones((xs.shape[0],), dtype=np.float) / xs.shape[0]


# Return a distribution over the times such that the newest time has weight
# 1/2, the second newest has 1/4 and so on.
def time_simple(xs):
    weights = [1.0 / (2.0**x) for x in range(1, len(xs))]
    if len(weights) == 0:
        weights = [1.0]
    else:
        weights.append(weights[-1])
    sort = np.flip(np.argsort(xs), 0)

    return np.array(weights)[sort]


# Weight by time in seconds such that the newest text is given the highest
# weight. The oldest text will have weight 0.
def time_weighted(xs):
    weights = xs - np.min(xs)
    weights = weights / np.sum(weights)

    return weights


# Weight by time monthly and arbitrarily add one to all of them to give the
# oldest text another weight than 0.
def time_weighted_2(xs):
    seconds_per_month = 2629743
    xs = xs / seconds_per_month

    weights = xs - np.min(xs) + 1
    weights = weights / np.sum(weights)

    return weights


months = [480.60179265, 483.69015527, 487.92844016, 488.15842461, 490.52397896,
          493.05380792, 495.87933117]
month_labels = list(map(lambda x: int(x), months))
months = np.array(list(map(lambda x: x * 2629743, months)))

uniform_weights = uniform(months)
time_simple_weights = time_simple(months)
time_weighted_weights = time_weighted(months)
time_weighted_2_weights = time_weighted_2(months)

width = 0.2
ind = np.arange(len(months))

fig, ax = plt.subplots()
rects1 = ax.bar(ind - 1*width, uniform_weights, width, color='r', label='Uniform')
rects2 = ax.bar(ind - 0*width, time_simple_weights, width, color='b', label='Time Weighted Simple')
rects3 = ax.bar(ind + 1*width, time_weighted_weights, width, color='y', label='Time Weighted Relative')
rects4 = ax.bar(ind + 2*width, time_weighted_2_weights, width, color='g', label='Time Weighted Relative Monthly')

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(map(lambda x: 'Month {}'.format(x), month_labels))
ax.grid(True)
ax.set_ylabel('Weight')

ax.legend()
# ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

plt.show()
