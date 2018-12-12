# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt
@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))



def build_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]

    return sparse.coo_matrix((training_ratings, (rows, cols))).tocsr()


def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameterssklearn.ensemble.AdaBoostRegressor
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    # Feature for users
    rating_matrix = rating_matrix.tocsr()
    user_features = rating_matrix[user_movie_pairs[:, 0]]


    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()
    X = sparse.hstack((user_features, movie_features))
    return X.tocsr()


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()


def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name


def adaboost():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    df = pd.read_csv(os.path.join(prefix, 'train_user_movie_merge.csv'), delimiter=',',dtype=float)

    train_features = df.columns
    training_with_more_features = df.values.squeeze()  

    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # Best estimator after hyperparameter tuning
    base_model = RandomForestRegressor(bootstrap=True, n_estimators=38, criterion='mse',verbose=0)

    model =  AdaBoostRegressor(base_model)
    print(model)
    with measure_time('Training'):
        model.fit(X_train, y_train)
    feature_importances = pd.DataFrame(model.feature_importances_,
                                    index = train_features,
                                    columns=['importance']).sort_values('importance',ascending=False)

    print(feature_importances)
    #Check for overfitting
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    print("Test set MSE: {}".format(MSE_test))
    print("Train set MSE: {}".format(MSE_train))

    # #Plot accuracy for different max_depths
    # print(accuracies)
    # plt.plot(maxdepths,accuracies)
    # plt.xlabel("maxdepths")
    # plt.ylabel("mean_squared_error")
    # plt.savefig("RandomForest_precise.svg")

    # -----------------------Submission: Running model on provided test_set---------------------------- #

    #Load test data
    X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    X_test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    #Predict
    print("Predicting...")
    y_pred = model.predict(X_test)

    fname = make_submission(y_pred, X_test_user_movie_pairs, 'AdaboostWithRandomForest')
    print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    adaboost()
    # check_overfitting()