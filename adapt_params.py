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

# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

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

    Parameters
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




def decisiontreemethod():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_train = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    y_train = load_from_csv(os.path.join(prefix, 'output_train.csv'))


    # X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # user_movie_rating_triplets = np.hstack((X_train,
    #                                         y_train.reshape((-1, 1))))

    # # Build the learning matrix
    # rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    # X_train_matrix = create_learning_matrices(rating_matrix, X_train)




    # maxdepths = list(range(1,10,1))
    # cv_results = []
    # for maxdepth in maxdepths:
    #     filename = "DTR_maxd_{}.pkl".format(maxdepth)

    #     #Skip if the model has already been trained at this depth
    #     if(os.path.isfile(filename)):
    #         print("Model with depth {} already trained. Import filename {}".format(maxdepth, filename))
    #         continue

    #     model = DecisionTreeRegressor(max_depth = maxdepth)
    #     start = time.time()
    #     with measure_time('Training'):
    #         print('Training...with a max_depth of {}'.format(maxdepth))
    #         scores = cross_val_score(model, training_with_more_features, training_labels, cv=5, scoring='neg_mean_squared_error')
    #         print(scores)
    #         cv_results.append(scores.mean()) # Taking the mean of the cv_val tries
    #         # model.fit(X_train, y_train)
            

    #     #Save estimator to file so that we train once
    #     joblib.dump(model, filename) 

    # #Unconstrained model
    # #Skip if the model has already been trained at this depth
    # if(os.path.isfile("DTR_None.pkl")):
    #     print("Model with no depth constrained already trained. Import filename DTR_None.pkl")
    # else:
    #     start = time.time()
    #     with measure_time('Training'):
    #         print('Training...with no max_depth')
    #         model = DecisionTreeRegressor().fit(X_ls, y_ls)
    #     #Save estimator to file so that we train once
    #     joblib.dump(model, "DTR_None.pkl") 

    # # Importing estimators from filename
    # models = []
    # for maxdepth in maxdepths:
    #     filename = "DTR_maxd_{}.pkl".format(maxdepth)
    #     print("Loading estimator {}".format(filename))
    #     models.append(joblib.load(filename))

    # #Importing unconstrained model
    # print("Loading estimator DTR_None.pkl".format(filename))
    # models.append(joblib.load("DTR_None.pkl"))

    # # ---------Prediction - Selecting best parameters---------------------------- #


    # # Build the prediction matrix
    # user_movie_rating_triplets = np.hstack((X_test,
    #                                         y_test.reshape((-1, 1))))
    # rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    # X_ts_matrix = create_learning_matrices(rating_matrix, X_test)
    

    # Predict
    # accuracies = []
    # for model in models:
    #     print("Predicting...")
    #     y_pred = model.predict(X_test)
    #     accuracies.append(mean_squared_error(y_test, y_pred))
    
    # # #Getindex for unconstrained depth
    # # maxdepths.append(100)
    # #Plot accuracy for different max_depths
    # print(accuracies)
    # plt.plot(maxdepths,accuracies)
    # plt.xlabel("maxdepths")
    # plt.ylabel("mean_squared_error")
    
    # plt.show()
    # plt.savefig("MSE_DT.svg")

    # print(cv_results)
    # plt.plot(maxdepths,cv_results)
    # plt.xlabel("maxdepths")
    # plt.ylabel("neg_mean_squared_error, mean over 5 iterations")
    
    # plt.show()
    # plt.savefig("MSE_DT_Crossval5.svg")


    # # ---------Submission: Running model on provided test_set---------------------------- #

    #Load test data
    X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    X_test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))
    #Predict
    print("Predicting...")
    y_pred = DecisionTreeRegressor(max_depth = 5).fit(X_train,y_train).predict(X_test)

    fname = make_submission(y_pred, X_test_user_movie_pairs, 'DTR_5')
    print('Submission file "{}" successfully written'.format(fname))


def knrmethod():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                            'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_user_movie_pairs, training_labels, test_size=0.2, random_state=42)

    user_movie_rating_triplets = np.hstack((X_train,
                                            y_train.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    X_ls = create_learning_matrices(rating_matrix, X_train)

    y_ls = y_train

    neighbors = list(range(1,60,5))
    for neighbor in neighbors:
        filename = "KNR_neigbors_{}.pkl".format(neighbor)

        #Skip if the model has already been trained at this depth
        if(os.path.isfile(filename)):
            print("Model with depth {} already trained. Import filename {}".format(neighbor, filename))
            continue

        model = KNeighborsRegressor(n_neighbors = neighbor)
        start = time.time()
        with measure_time('Training'):
            print('Training...with a n_neighbors of {}'.format(neighbor))
            model.fit(X_ls, y_ls)

        #Save estimator to file so that we train once
        joblib.dump(model, filename) 

    # Importing estimators from filename
    models = []
    for neighbor in neighbors:
        filename = "KNR_neigbors_{}.pkl".format(neighbor)
        print("Loading estimator {}".format(filename))
        models.append(joblib.load(filename))

    # ---------Prediction - Selecting best parameters---------------------------- #


    # Build the prediction matrix
    user_movie_rating_triplets = np.hstack((X_test,
                                            y_test.reshape((-1, 1))))
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    X_ts_matrix = create_learning_matrices(rating_matrix, X_test)

    y_ts = y_test

    # Predict
    # accuracies = []
    # for model in models:
    #     print("Predicting...")
    #     y_pred = model.predict(X_ts_matrix)
    #     accuracies.append(mean_squared_error(y_ts, y_pred))

    # #Plot accuracy for different max_depths
    # print(accuracies)
    # plt.plot(neighbors,accuracies)
    # plt.xlabel("n_neighbors")
    # plt.ylabel("mean_squared_error")
    
    # plt.show()
    # plt.savefig("MSE_KNR.svg")

    # ---------Submission: Running model on provided test_set---------------------------- #
    
    print("Predicting...")
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))
    # Build the prediction matrix
    X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)
    #Predict
    y_pred = models[-1].predict(X_ts)

    fname = make_submission(y_pred, test_user_movie_pairs, 'KNR_56')
    print('Submission file "{}" successfully written'.format(fname))


if __name__ == '__main__':
   
    decisiontreemethod() # Kaggle error of 1.27
    
    # knrmethod() # Kaggle error of 2.56
    

    # ------------------------------ Prediction ------------------------------ #
    # # Load test data
    # test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # # Build the prediction matrix
    # X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)

    # # Predict
    # y_pred = model.predict(X_ts)

    # # Making the submission file
    # fname = make_submission(y_pred, test_user_movie_pairs, 'toy_example')
    # print('Submission file "{}" successfully written'.format(fname))
