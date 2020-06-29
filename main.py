import sys
from data import Data
from algorithm_runner import AlgorithmRunner
from sklearn.neighbors import KNeighborsClassifier


def main(argv):
    path = "movie_metadata.csv"
    movies_data = Data(path)
    movies_data.preprocess()
    folds = movies_data.split_to_k_Folds()
    neigh = KNeighborsClassifier(n_neighbors=5)
    for fold in folds:
        neigh.fit(fold)
        print(neigh.predict(fold))

    knn = AlgorithmRunner(neigh)


if __name__ == '__main__':
    main(sys.argv)
