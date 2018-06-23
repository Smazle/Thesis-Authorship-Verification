# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle


class FeatureSearch:
    """
        Given a set of extracted training features and a classifier
        the goal of this class is to determine the best set of features
        for the given classifier, using forward greedy selection,
        and stratified cross validation.

        The class is initialized with a classifier, the minimum
        features one wants selected, how many authors to use,
        if the data should be normalized, and how many folds in cross
        validation should be used.

        Attributes:
            data (Numpy Matrix): Extracted training features
            authors (List): List of limited authors selected

    """

    data = None

    authors = None

    def __init__(self,
                 classifier,
                 minFeatureCount,
                 authorLimit=None,
                 normalize=True,
                 validator=3):

        self.classifier = classifier
        self.minFeatureCount = minFeatureCount
        self.authorLimit = authorLimit
        self.normalize = normalize
        self.validator = validator
        self.scaler = None

    def fit(self, dataFile, outfile):
        self.__generateData__(dataFile)
        print('Unique authors', len(np.unique(self.authors)))
        print('Training Data Shape', self.data.shape)
        print(sorted(np.unique(self.authors)))
        with open(outfile, 'w') as f:
            for feature, value in self.feature_generator():
                print('Feature Selected', feature)
                f.write(str(feature) + ', ' + str(value) + '\n')
                f.flush()

    def feature_generator(self):
        # Initialize selected features.
        selectedFeatures = []
        missingFeatures = list(range(self.maxFeatureCount))

        # Loop over the minimum amount of features we want.
        for count in range(self.minFeatureCount):

            maxIdx = maxVal = 0

            # Loop over the different features.
            for feature_idx in missingFeatures:
                currentFeatures = selectedFeatures + [feature_idx]

                score = self.__evaluate_classifier__(self.classifier,
                                                     currentFeatures)

                print(feature_idx, score)

                # If the average over all authors for that features is is
                # better, replace the former max value/idx.
                if score > maxVal:
                    maxIdx = feature_idx
                    maxVal = score

            selectedFeatures.append(maxIdx)
            missingFeatures.remove(maxIdx)
            yield maxIdx, maxVal

    # Get the average score per author for the current features.
    def __evaluate_classifier__(self, classifier, features):
        authorScores = []
        for author in np.unique(self.authors):
            X, y = self.__generateAuthorData__(author)

            score = cross_val_score(
                classifier, X[:, features], y, cv=self.validator)

            authorScores.append(np.mean(score))

        return np.mean(authorScores)

    def __generateData__(self, filePath):
        with open(filePath, 'r') as f:
            data = pd.read_csv(f)
            print(data.columns.values)
            self.authors = data.as_matrix(columns=['author']).flatten()

            datacols = filter(lambda x: x != 'author', data.columns)
            self.data = data.as_matrix(columns=datacols)

        if self.authorLimit is not None:
            unique_authors = np.unique(self.authors)

            unique_authors = np.array([
                x for x in unique_authors
                if len(self.data[self.authors == x]) > 3
            ])

            np.random.shuffle(unique_authors)
            unique_authors = \
                unique_authors[:int(len(unique_authors) * self.authorLimit)]
            self.data = self.data[np.isin(self.authors,
                                          unique_authors)].astype(np.float)
            self.authors = self.authors[np.isin(self.authors,
                                                unique_authors)].astype(np.int)

        if self.normalize:
            scaler = StandardScaler()
            scaler.fit(self.data)
            pickle.dump(scaler, open('Scaler.p', 'wb'))
            self.data = scaler.transform(self.data)
            self.scaler = scaler

        self.maxFeatureCount = self.data.shape[1]

    def __generateAuthorData__(self, author):

        # Fetch own texts
        own_texts = self.data[self.authors == author]

        # Fetch other texts
        enum_authors = enumerate(self.authors)
        other_texts = list(filter(lambda x: x[1] != author, enum_authors))

        # Shuffle and pick same number of texts a len(own_texts)
        np.random.shuffle(other_texts)
        other_texts = [x[0] for x in other_texts[:len(own_texts)]]

        # Extract the actual texts using the idx
        other_texts = self.data[other_texts]

        X = np.append(own_texts, other_texts, axis=0)
        y = np.array([1] * len(own_texts) + [0] * len(other_texts))

        return X, y
