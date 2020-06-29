import sys
from data import Data
from algorithm_runner import AlgorithmRunner
from sklearn.neighbors import KNeighborsClassifier


def main(argv):
    path = "movie_metadata.csv"
    movies_data = Data(path)
    movies_data.preprocess()
    fold1, fold2, fold3, fold4, fold5 = movies_data.split_to_k_Folds()
    folds = [fold1, fold2, fold3, fold4, fold5]
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    Knn_algo = AlgorithmRunner(knn_classifier)
    for fold in folds:
        train_bin = []
        test_bin = []
        train_df = movies_data.df.iloc[fold[0]]
        test_df = movies_data.df.iloc[fold[1]]
        for i in range(len(movies_data.labels)):
            if i in fold[0]:
                train_bin.append(movies_data.labels[i])
            else:
                test_bin.append(movies_data.labels[i])
        Knn_algo.fit(train_df, train_bin)
        knn_predict = Knn_algo.predict(test_df)
        print (knn_predict)


if __name__ == '__main__':
    main(sys.argv)
