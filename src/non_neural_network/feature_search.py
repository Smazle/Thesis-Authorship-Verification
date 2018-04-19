# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score, LeaveOneOut
import numpy as np
import pandas as pd


class FeatureSearch:

    xTrain = None

    authors = None

    def __init__(self, classifiers, minFeatureCount, authorLimit=None):
        self.classifiers = classifiers
        self.minFeatureCount = minFeatureCount
        self.authorLimit = authorLimit

    def fit(self, dataFile, outfile):
        print('Starting Feature Search')
        self.__generateData__(dataFile)
        with open(outfile, 'w') as f:
            for feature in self.feature_generator():
                print('Feature Selected', selectedFeatures)
                f.write(str(feature) + ', ')

    def feature_generator(self,):

        # Loop over supplied classifiers.
        for classifier in self.classifiers:

            # Initialize selected features.
            selectedFeatures = []
            missingFeatures = list(range(self.maxFeatureCount))

            # Loop over the minimum amount of features we want.
            for count in range(self.minFeatureCount):

                maxIdx = maxVal = 0

                # Loop over the different features.
                for feature_idx in missingFeatures:
                    currentFeatures = selectedFeatures + [feature_idx]

                    score = self.__evaluate_classifier__(
                        classifier, currentFeatures)

                    print(feature_idx, score)

                    # If the average over all authors for that features is is
                    # better, replace the former max value/idx.
                    if score > maxVal:
                        maxIdx = feature_idx
                        maxVal = score

                selectedFeatures.append(maxIdx)
                missingFeatures.remove(maxIdx)
                yield maxIdx

    # Get the average score per author for the current features.
    def __evaluate_classifier__(self, classifier, features):
        authorScores = []
        for author in np.unique(self.authors):
            X, y = self.__generateAuthorData__(author)

            score = cross_val_score(
                classifier, X[:, features], y, cv=LeaveOneOut(), n_jobs=-1)

            authorScores.append(np.mean(score))

        return np.mean(authorScores)

    def __generateData__(self, filePath):
        with open(filePath, 'r') as f:
            data = pd.read_csv(f)
            self.authors = data.as_matrix(columns=['author']).flatten()

            datacols = filter(lambda x: x != 'author', data.columns)
            self.xTrain = data.as_matrix(columns=datacols)

        if self.authorLimit is not None:
            unique_authors = np.unique(self.authors)
            np.random.shuffle(unique_authors)
            unique_authors = unique_authors[:self.authorLimit]
            self.xTrain = self.xTrain[np.isin(self.authors, unique_authors)]
            self.authors = self.authors[np.isin(self.authors, unique_authors)]

        self.maxFeatureCount = self.xTrain.shape[1]

    def __generateAuthorData__(self, author):

        # Fetch own texts
        own_texts = self.xTrain[self.authors == author]

        # Fetch other texts
        enum_authors = enumerate(self.authors)
        other_texts = list(filter(lambda x: x[1] != author, enum_authors))

        # Shuffle and pick same number of texts a len(own_texts)
        np.random.shuffle(other_texts)
        other_texts = [x[0] for x in other_texts[:len(own_texts)]]

        # Extract the actual texts using the idx
        other_texts = self.xTrain[other_texts]

        X = np.append(own_texts, other_texts, axis=0)
        y = np.array([1] * len(own_texts) + [0] * len(other_texts))

        return X, y
