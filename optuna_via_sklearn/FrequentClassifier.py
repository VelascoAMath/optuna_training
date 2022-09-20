from sklearn.base import ClassifierMixin
from sklearn.linear_model._base import BaseEstimator
import numpy as np
import random
from collections import Counter

class FrequentClassifier(ClassifierMixin, BaseEstimator):
    """
    Frequent classifier. Returns the most frequent label in the training data
    """


    def fit(self, X, Y):
        self._num_inputs = X.shape[1]
        self._c = Counter()
        for y in Y:
            if y != 0 and y != 1:
                raise Exception(f"The labels({y}) must be 0 or 1! It contains {y}")
            self._c.update([y])

        self._most_common_label = self._c.most_common(1)[0][0]

        if self._most_common_label == 0:
            self._prob_list = [1, 0]
        elif self._most_common_label == 1:
            self._prob_list = [0, 1]
        else:
            raise Exception(f"The labels contain invalid items! They should only contain 0s and 1s! {self._c}")

    def predict(self, X):
        
        if X.shape[1] != self._num_inputs:
            raise Exception(f"The shape of the input {X.shape} needs to match (_, {self._num_inputs})!")

        y = [self._most_common_label for x in X]

        return np.array(y)


    def predict_proba(self, X):
        
        if X.shape[1] != self._num_inputs:
            raise Exception(f"The shape of the input {X.shape} needs to match (_, {self._num_inputs})!")

        y = [self._prob_list for x in X]

        return np.array(y)


def main():

    clf = FrequentClassifier()

    X = np.random.randint(-5, 5, (50, 4))
    Y = np.array(random.choices([0, 1], weights=[0.7, 0.3], k=50))

    clf.fit(X, Y)

    print(clf.predict(X))

if __name__ == '__main__':
    main()
