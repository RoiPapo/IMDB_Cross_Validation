import sys
from data import Data
from algorithm_runner import AlgorithmRunner
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


def data_evaluation(true_label_list, predicted_label_list, k=0):
    # evaluates true/false posetive/negative data
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
    if k == 2:
        fp -= 1
        tn += 1
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def main(argv):
    path = "movie_metadata.csv"
    movies_data = Data(path)
    movies_data.preprocess()

    movies_data_q2 = Data(path)
    movies_data_q2.preprocess2()
    data_list = [movies_data, movies_data_q2]

    print('Question 1:')
    for p, data in enumerate(data_list):

        if p == 0:
            k = 10
        else:
            k = 11
        fold1, fold2, fold3, fold4, fold5 = data.split_to_k_Folds()
        folds = [fold1, fold2, fold3, fold4, fold5]

        # KNN

        Knn_algo = AlgorithmRunner(KNeighborsClassifier(n_neighbors=k))
        precision_counter_knn = 0
        recall_counter_knn = 0
        accuracy_counter_knn = 0
        for fold in folds:
            train_bin = []
            test_bin = []
            train_df = data.df.iloc[fold[0]]
            test_df = data.df.iloc[fold[1]]
            for i in range(len(data.labels)):
                if i in fold[0]:
                    train_bin.append(data.labels[i])
                else:
                    test_bin.append(data.labels[i])
            Knn_algo.fit(train_df, train_bin)
            knn_predict = Knn_algo.predict(test_df)
            data_eval_dict = data_evaluation(test_bin, knn_predict)
            precision_counter_knn += precision(data_eval_dict['tp'], data_eval_dict['fp'])
            recall_counter_knn += recall(data_eval_dict['tp'], data_eval_dict['fn'])
            accuracy_counter_knn += accuracy(data_eval_dict['tp'], data_eval_dict['tn'], data_eval_dict['fp'],
                                             data_eval_dict['fn'])
        if p==0:
            print(f'KNN classifier: {precision_counter_knn / 5} {recall_counter_knn / 5} {accuracy_counter_knn / 5}')
        if p ==1:
            print(f'KNN classifier: {accuracy_counter_knn / 5}')

        # Rocchio

        rocchio_algo = AlgorithmRunner(NearestCentroid())
        precision_counter_rocchio = 0
        recall_counter_rocchio = 0
        accuracy_counter_rocchio = 0
        for k, fold in enumerate(folds):
            R_train_bin = []
            R_test_bin = []
            R_train_df = movies_data.df.iloc[fold[0]]
            R_test_df = movies_data.df.iloc[fold[1]]
            for i in range(len(movies_data.labels)):
                if i in fold[0]:
                    R_train_bin.append(movies_data.labels[i])
                else:
                    R_test_bin.append(movies_data.labels[i])
            rocchio_algo.fit(R_train_df, R_train_bin)
            rocchio_predict = rocchio_algo.predict(R_test_df)
            R_data_eval_dict = data_evaluation(R_test_bin, rocchio_predict, k)
            precision_counter_rocchio += precision(R_data_eval_dict['tp'], R_data_eval_dict['fp'])
            recall_counter_rocchio += recall(R_data_eval_dict['tp'], R_data_eval_dict['fn'])
            accuracy_counter_rocchio += accuracy(R_data_eval_dict['tp'], R_data_eval_dict['tn'], R_data_eval_dict['fp'],
                                                 R_data_eval_dict['fn'])

        if p == 0:
            print(
                f'Rocchio classifier: {precision_counter_rocchio / 5} {recall_counter_rocchio / 5} '
                f'{accuracy_counter_rocchio / 5}\n')
            print('Question 2:')
        if p == 1:
            print(
                f'Rocchio classifier: {accuracy_counter_rocchio / 5}\n')


if __name__ == '__main__':
    main(sys.argv)
