from sklearn.neighbors import KNeighborsClassifier


class AlgorithmRunner:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self):
        self.algorithm.fit()

    def predict(self):
        self.algorithm.predict()
