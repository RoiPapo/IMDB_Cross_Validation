from sklearn.neighbors import KNeighborsClassifier


class AlgorithmRunner:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self, train_set, labels):
        self.algorithm.fit(train_set, labels)

    def predict(self, train_set):
        return self.algorithm.predict(train_set).tolist()
