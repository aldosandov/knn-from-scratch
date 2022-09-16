from math import sqrt
from functools import reduce
from collections import Counter
import numpy as np


class KnnClassifier():
    def __init__(self, n_neighbors=3, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def __distance(self, a, b):
        if a.shape[0] != b.shape[0]:
            raise ValueError("Distance: the shape of points doesn't match.")

        dim = a.shape[0]

        if self.metric == "euclidean":
            dist = sqrt(reduce(lambda i, j: i + ((a[j] - b[j]) ** 2), range(dim), 0))
            return dist
        else: # Manhattan distance.
            dist = sum(abs(i - j) for i, j in zip(a, b))
            return dist


    def fit(self, x_train, y_train):
        """ Fit the classifier.

        Arguments:
            x_train {numpy.ndarray} -- Predictors data for training.
            y_train {numpy.ndarray} -- Target data for training.
        """
        self.__points = x_train
        self.__classes = y_train
        self.__dim = x_train[0].shape[0]


    def predict(self, x_test):
        """ Predict the class of the points given.

        Arguments:
            x_test {numpy.ndarray} -- Data for prediction.

        Raises:
            ValueError: train and test dataset shapes are different.

        Returns:
            y_pred {numpy.ndarray} -- Prediction data.
        """
        
        y_pred = []

        if x_test[0].shape[0] != self.__dim:
            raise ValueError("Predict: train and test dataset shapes are different.")
        
        for x in x_test: # Each point of x_test.
            distances = []

            # Calculating the distance to all training points.
            for idx, point in enumerate(self.__points):
                dist = self.__distance(x, point)
                group = self.__classes[idx]

                distances.append((dist, group))

            # Determining the most common class of neighbors.
            distances.sort()
            neighbors = distances[:self.n_neighbors]
            groups = [e[1] for e in neighbors]
            target = Counter(groups).most_common()[0][0]
            y_pred.append(target)

        return np.array(y_pred)

    def score(self, y_pred, y_true):
        """ Model scoring

        Arguments:
            y_pred {nunpy.ndarray} -- Data of the predicted class of the points.
            y_true {nunpy.ndarray} -- Data of the real class of the points.

        Raises:
            ValueError: arrays shape does't match

        Returns:
            score {float} -- Score of the model.
        """
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("Score: arrays shape does't match.")
        
        dim = len(y_pred)

        true_pos = 0
        for a, b in zip(y_pred, y_true):
            if a == b:
                true_pos += 1

        score = true_pos / dim

        return score