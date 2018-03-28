#!/usr/bin/env python3

import argparse
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Set seed to make sure we get reproducible results.
np.random.seed(7)


parser = argparse.ArgumentParser(
    description='Find best features from feature file via a greedy search')
parser.add_argument(
    'features',
    type=str,
    help='Path to file containing features'
)
args = parser.parse_args()

with open(args.features, 'r') as feature_file:
    feature_file.readline()  # Skip first line.
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    # Number of features is number of columns minus the author column.
    feature_n = len(reader.__next__()) - 1

    line_n = 1
    for line in reader:
        line_n = line_n + 1

    X = np.zeros((line_n, feature_n), dtype=np.float)
    authors = np.zeros((line_n, ), dtype=np.int)

    # Go back to start of file and read again.
    feature_file.seek(0)
    feature_file.readline()
    reader = csv.reader(feature_file, delimiter=' ', lineterminator='\n')

    for i, line in enumerate(reader):
        X[i] = np.array(list(map(lambda x: float(x), line[0:-1])))
        authors[i] = int(line[-1])

unique_authors = np.sort(np.unique(authors))
print(unique_authors.shape)

# use_these = np.array([
          # 0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,
         # 11,  300,  301,  302,  303,  304,  305,  306,  307,  308,  309,
        # 310,  311,  600,  601,  602,  603,  604,  605,  606,  607,  608,
        # 609,  610,  611,  900,  901,  902,  903,  904,  905,  906,  907,
        # 908,  909,  910,  911, 1200, 1201, 1202, 1203, 1204, 1205, 1206,
       # 1207, 1208, 1209, 1210, 1211, 1500, 1501, 1502, 1503, 1504, 1505,
       # 1506, 1507, 1508, 1509, 1510, 1511, 1800, 1801, 1802, 1803, 1804,
       # 1805, 1806, 1807, 1808, 1809, 1810, 1811, 2100, 2101, 2102, 2103,
       # 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2400, 2401, 2402,
       # 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2700, 2701,
       # 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2750,
       # 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761,
       # 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810,
       # 2811, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859,
       # 2860, 2861, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358,
       # 3359, 3360, 3361, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407,
       # 3408, 3409, 3410, 3411, 3450, 3451, 3452, 3453, 3454, 3455, 3456,
       # 3457, 3458, 3459, 3460, 3461, 3950, 3951, 3952, 3953, 3954, 3955,
       # 3956, 3957, 3958, 3959, 3960, 3961, 4450, 4451, 4452, 4453, 4454,
       # 4455, 4456, 4457, 4458, 4459, 4460, 4461], dtype=np.int)

# X = X[:, use_these]

# scores = []
# for author in unique_authors:
    # print('Testing author', author)
    # author_texts = X[authors == author]
    # other_texts = X[authors != author]

    # opposition = other_texts[np.random.choice(
        # other_texts.shape[0],
        # author_texts.shape[0],
        # replace=False)]

    # # TODO: Change C and gamma values.
    # classifier = SVC(kernel='rbf', C=100, gamma=0.00001)
    # X_train = np.vstack([author_texts, opposition])
    # y_train = np.array([1] * author_texts.shape[0] + [0] * author_texts.shape[0])

    # score = cross_val_score(classifier, X_train, y_train)

    # scores.append(np.mean(score))

# print(scores)
# print(np.mean(scores))
