from sklearn.base import ClassifierMixin
from sklearn.linear_model._base import BaseEstimator
import numpy as np

class RandomClassifier(ClassifierMixin, BaseEstimator):
    """
    Random classifier. Returns random values
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self._num_inputs = X.shape[1]

    def predict(self, X):
        
        if X.shape[1] != self._num_inputs:
            raise Exception(f"The shape of the input {X.shape} needs to match (_, {self._num_inputs})!")

        if self.random_state is not None:
            np.random.seed(self.random_state)


        y = [np.random.randint(0, 2) for x in X]

        return np.array(y)

    def predict(self, X):
        
        if X.shape[1] != self._num_inputs:
            raise Exception(f"The shape of the input {X.shape} needs to match (_, {self._num_inputs})!")

        if self.random_state is not None:
            np.random.seed(self.random_state)


        y = [np.random.randint(0, 2) for x in X]

        return np.array(y)

    def predict_proba(self, X):
        
        if X.shape[1] != self._num_inputs:
            raise Exception(f"The shape of the input {X.shape} needs to match (_, {self._num_inputs})!")

        if self.random_state is not None:
            np.random.seed(self.random_state)


        y = []
        for x in X:
            p = np.random.rand()
            y.append([p, 1 - p])

        return np.array(y)


def main():

    clf = RandomClassifier(random_state=7)

    X = np.random.randint(-5, 5, (50, 4))
    Y = np.random.randint(0, 2, 50)

    clf.fit(X, Y)

    print(clf.predict(X))

if __name__ == '__main__':
    main()
