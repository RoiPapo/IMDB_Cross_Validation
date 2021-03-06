import sklearn
import sklearn.model_selection
import pandas as pd
import numpy as np


class Data:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.labels = None

    def preprocess(self):
        df = pd.read_csv(self.path)
        df = df.drop(["content_rating", "movie_imdb_link", "plot_keywords"], axis=1)
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.drop_duplicates(subset='movie_title', keep="first")
        # rename actor 1 2 and 3 to actor
        df = df.rename(columns={'actor_1_name': 'actor', 'actor_3_name': 'actor', 'actor_2_name': 'actor'})
        df = pd.get_dummies(df, columns=['actor', 'color', 'country',
                                         'director_name', 'language'], drop_first=False)
        cols_for_norm = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'aspect_ratio',
                         'budget', 'cast_total_facebook_likes', 'director_facebook_likes', 'duration',
                         'facenumber_in_poster', 'gross', 'movie_facebook_likes', 'num_critic_for_reviews',
                         'num_user_for_reviews', 'num_voted_users', 'title_year']

        # concat to my table

        a = np.array(df['imdb_score'].values.tolist())
        df['imdb_score'] = np.where(a >= 7, 1, 0).tolist()
        self.labels = df['imdb_score'].to_numpy()
        df = df.drop(["imdb_score"], axis=1)
        df = df.groupby(df.columns, axis=1).sum()  # sum of actrors
        genres = df["genres"].str.get_dummies(sep="|")
        df = df.drop(["genres"], axis=1)
        for col in cols_for_norm:
            df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df = pd.concat([df, genres], axis=1)
        df = df.drop(["movie_title"], axis=1)
        self.df = df




    def preprocess2(self):
        df = pd.read_csv(self.path)
        df = df.drop(["content_rating", "movie_imdb_link", "plot_keywords"], axis=1)
        df.replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.drop_duplicates(subset='movie_title', keep="first")
        # rename actor 1 2 and 3 to actor
        df = df.drop(["color"], axis=1)
        df = df.rename(columns={'actor_1_name': 'actor', 'actor_3_name': 'actor', 'actor_2_name': 'actor'})
        df = pd.get_dummies(df, columns=['actor',  'country',
                                         'director_name', 'language'], drop_first=False)
        cols_for_norm = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'aspect_ratio',
                         'budget', 'cast_total_facebook_likes', 'director_facebook_likes', 'duration',
                         'facenumber_in_poster', 'gross', 'movie_facebook_likes', 'num_critic_for_reviews',
                         'num_user_for_reviews', 'num_voted_users', 'title_year']

        # concat to my table

        a = np.array(df['imdb_score'].values.tolist())
        df['imdb_score'] = np.where(a >= 7, 1, 0).tolist()
        self.labels = df['imdb_score'].to_numpy()
        df = df.drop(["imdb_score"], axis=1)
        df = df.groupby(df.columns, axis=1).sum()  # sum of actrors
        genres = df["genres"].str.get_dummies(sep="|")
        df = df.drop(["genres"], axis=1)
        for col in cols_for_norm:
            df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df = pd.concat([df, genres], axis=1)
        df = df.drop(["movie_title"], axis=1)
        df = df.drop(["duration"], axis=1)
        df = df.drop(["facenumber_in_poster"], axis=1)
        self.df = df


    def split_to_k_Folds(self):
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)
        # nmpyarr = self.df.to_numpy()
        return kf.split(self.labels)
