import sys
from data import Data
from algorithm_runner import AlgorithmRunner
from sklearn.neighbors import KNeighborsClassifier


def data_evaluation(true_label_list, predicted_label_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(true_label_list)):
        if true_label_list[i] == predicted_label_list[i]:
            if true_label_list[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true_label_list[i] == 1:
                fn += 1
            else:
                fp += 1
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def precision(tp, fp):
    return (tp / (tp + fp))


def recall(tp, fn):
    return (tp / (tp + fn))

def calc_accuracy(test_set, classifier):
    correct = 0.0
    total = len(test_set.keys())
    for key in test_set:
        real = test_set[key][-1]
        predicted = classifier.predict(test_set[key][0:-1])
        if real == predicted:
            correct += 1.0
    return correct / total


def main(argv):
    path = "movie_metadata.csv"
    movies_data = Data(path)
    movies_data.preprocess()
    fold1, fold2, fold3, fold4, fold5 = movies_data.split_to_k_Folds()
    folds = [fold1, fold2, fold3, fold4, fold5]
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    Knn_algo = AlgorithmRunner(knn_classifier)
    precision_counter = 0
    recall_counter = 0
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
        data_eval_dict = data_evaluation(test_bin, knn_predict)
        precision_counter += precision(data_eval_dict['tp'], data_eval_dict['fp'])
        recall_counter += recall(data_eval_dict['tp'], data_eval_dict['fn'])
    print(f'Precision:{precision_counter / 5}, Recall: {recall_counter / 5}, Accuracy: {}')


if __name__ == '__main__':
    main(sys.argv)
