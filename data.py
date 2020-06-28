import sklearn


def preprocess():
    pass


def split_to_k_Folds():
    sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)
