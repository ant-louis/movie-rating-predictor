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
    training_merge_data = load_from_csv(os.path.join(prefix, 'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_merge_data, training_labels, test_size=0.1, random_state=42)
    
    #Tuning the complexity of the DecisionTree
    maxdepths = list(range(1,X_train.shape[1],1))
    cv_results = []
    models = []
    for maxdepth in maxdepths:
        model = DecisionTreeRegressor(max_depth = maxdepth)
        with measure_time('Training'):
            print('Training...with a max_depth of {}'.format(maxdepth))
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
            print(scores)
            cv_results.append(scores.mean()) # Taking the mean of the cv_val tries
            model.fit(X_train, y_train)
        models.append(model)

    # # ---------Selecting best parameters when building different models---------------------------- #
    """ Needs a test/train split and different models"""

    accuracies = []
    i = 0
    for model in models:
        print("Predicting...")
        y_pred = model.predict(X_test)
        accuracy = mean_squared_error(y_test, y_pred)
        accuracies.append(accuracy)
        print("Model: {} MSE: {}".format(i,accuracy))
        i+=1
      
     best_accuracy = min(accuracies)
     best_depth = accuracies.index(min(accuracies))
     print("Best accuracy: {} - Maxdepth: {}".format(best_accuracy, best_depth))

    
    #Plot accuracy for different max_depths
    print(accuracies)
    plt.plot(maxdepths,accuracies)
    plt.xlabel("maxdepths")
    plt.ylabel("mean_squared_error")
    
    # filename = ".svg"
    # plt.savefig(filename)

    # ---------Plotting cross-validation results---------------------------- #

    print(cv_results)
    plt.plot(maxdepths,cv_results)
    plt.xlabel("maxdepths")
    plt.ylabel("Negative_mean_squared_error")
    # plt.savefig("MSE_DT_features_Crossval10.svg")


    # # ---------Submission: Running model on provided test_set---------------------------- #

    # #Load test data
    # X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    # X_test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))
    # #Predict
    # print("Predicting...")
    # y_pred = DecisionTreeRegressor(max_depth = 8).fit(X_train,y_train).predict(X_test)

    # fname = make_submission(y_pred, X_test_user_movie_pairs, 'DTR_5')
    # print('Submission file "{}" successfully written'.format(fname))


# def knrmethod():
#     prefix = 'Data/'

#     # ------------------------------- Learning ------------------------------- #
#     # Load training data
#     training_merge_data = load_from_csv(os.path.join(prefix, 'train_user_movie_merge.csv'))
#     training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

#     X_train, X_test, y_train, y_test = train_test_split(training_merge_data, training_labels, test_size=0.1, random_state=42)

#     #Tuning the complexity of the KNRegressor
#     neighbors = list(range(1,201,5))
#     cv_results = []
#     models = []
#     for neighbor in neighbors:
#         model = KNeighborsRegressor(n_neighbors = neighbor)
#         with measure_time('Training'):
#             print('Training...with a n_neighbors of {}'.format(neighbor))
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
#             print(scores.mean())
#             cv_results.append(scores.mean()) # Taking the mean of the cv_val tries
#             model.fit(X_train, y_train)
#         models.append(model)

#     # # ---------Selecting best parameters when building different models---------------------------- #
#     """ Needs a test/train split and different models"""

#     accuracies = []
#     i = 1
#     for model in models:
#         print("Predicting...")
#         y_pred = model.predict(X_test)
#         accuracy = mean_squared_error(y_test, y_pred)
#         accuracies.append(accuracy)
#         print("Model: {} MSE: {}".format(i,accuracy))
#         i+=5

#     best_accuracy = min(accuracies)
#     best_neigh = accuracies.index(min(accuracies))
#     print("Best accuracy: {} - Nb of neighbors: {}".format(best_accuracy, best_neigh))
    
#     # #Plot accuracy for different n_neighbors
#     # print(accuracies)
#     # plt.plot(neighbors,accuracies)
#     # plt.xlabel("maxdepths")
#     # plt.ylabel("mean_squared_error")
    
#     # filename = ".svg"
#     # plt.savefig(filename)

#     # # # ---------Plotting cross-validation results---------------------------- #

#     # print(cv_results)
#     # plt.plot(neighbors,cv_results)
#     # plt.xlabel("n_neighbors")
#     # plt.ylabel("Negative_mean_squared_error")
#     # plt.savefig("NMSE_KNN_features_Crossval5.svg")

#     # ---------Submission: Running model on provided test_set---------------------------- #
    
#     # print("Predicting...")
#     # # Load test data
#     # test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))
#     # # Build the prediction matrix
#     # X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)
#     # #Predict
#     # y_pred = model.predict(X_ts)

#     # fname = make_submission(y_pred, test_user_movie_pairs, 'KNR_56')
#     # print('Submission file "{}" successfully written'.format(fname))



# def randomforest():
#     prefix = 'Data/'

#     # ------------------------------- Learning ------------------------------- #
#     # Load training data
#     training_with_more_features = load_from_csv(os.path.join(prefix,
#                                                             'train_user_movie_merge.csv'))
#     training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

#     X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

#     #Tuning the complexity(only max_depth) of the RandomForest
#     maxdepths = list(range(28,35,1))
#     for maxdepth in maxdepths:
#         filename = "estimators/RandomForest_maxd_{}.pkl".format(maxdepth)

#         #Skip if the model has already been trained at this depth
#         if(os.path.isfile(filename)):
#             print("RandomforestModel with max_d {} already trained. Import filename {}".format(maxdepth,filename))
#         else:
#             start = time.time()
#             with measure_time('Training'):
#                 model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=maxdepth, random_state=42,n_estimators=100, oob_score=True,n_jobs=1)
#                 print('Training...randomforest')
#                 model.fit(X_train, y_train)
#                 print("Maxdepth : {} Score : {}".format(maxdepth, model.oob_score_))

#                 #Save estimator to file so that we train once
#                 joblib.dump(model, filename) 
    
#     # # ---------Selecting best parameters when building different models---------------------------- #

#     models = []
#     # Importing estimators from filename
#     for maxdepth in maxdepths:
#         filename = "estimators/RandomForest_maxd_{}.pkl".format(maxdepth)
#         print("Loading estimator {}".format(filename))
#         if(os.path.isfile(filename)):
#             models.append(joblib.load(filename))
#         else:
#             break

#     # Predict
#     accuracies = []
#     i = 28
#     for model in models:
#         print("Predicting...")
#         y_pred = model.predict(X_test)
#         accuracy = mean_squared_error(y_test, y_pred)
#         accuracies.append(accuracy)
#         print("File: {} MSE: {}".format(i,accuracy))
#         i+=1
    
#     #Plot accuracy for different max_depths
#     print(accuracies)
#     plt.plot(maxdepths,accuracies)
#     plt.xlabel("maxdepths")
#     plt.ylabel("mean_squared_error")
#     plt.savefig("RandomForest_precise.svg")

#     # # ---------Submission: Running model on provided test_set---------------------------- #

#     # #Load test data
#     # X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
#     # X_test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

#     # models = joblib.load("RandomForest_maxd_31.pkl")

#     # #Predict
#     # print("Predicting...")
#     # y_pred = models.predict(X_test)

#     # fname = make_submission(y_pred, X_test_user_movie_pairs, 'RandomForr_31')
#     # print('Submission file "{}" successfully written'.format(fname))

def neuralNet():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_with_more_features = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    max_iters = list(range(5, 205, 5))
    for max_iter in max_iters:
        filename = "NNModel_maxiter_{}.pkl".format(max_iter)

        #Skip if the model has already been trained at this number of iterations
        if(os.path.isfile(filename)):
            print("NNModel with maxiter {} already trained. Import filename {}".format(max_iter,filename))
        else:
            start = time.time()
            with measure_time('Training'):
                model = MLPRegressor(hidden_layer_sizes = (100,), max_iter = max_iter, activation = 'logistic', random_state = 42)
                print('Training...neural net')
                model.fit(X_train, y_train)
                print("Max iterations : {} Loss : {}".format(max_iter, model.loss_))

                #Save estimator to file so that we train once
                joblib.dump(model, filename)

    models = []
    # Importing estimators from filename
    for max_iter in max_iters:
        filename = "NNModel_maxiter_{}.pkl".format(max_iter)
        print("Loading estimator {}".format(filename))
        if(os.path.isfile(filename)):
            models.append((joblib.load(filename), filename))
        else:
            break

    # Predict
    accuracies = []
    for model, filename in models:
        print("Predicting...")
        y_pred = model.predict(X_test)
        accuracy = mean_squared_error(y_test, y_pred)
        accuracies.append(accuracy)
        print("File: {} MSE: {}".format(filename,accuracy))
    
    length = len(models)
    #Plot accuracy for different max_depths
    print(accuracies)
    plt.plot(max_iters[:length],accuracies)
    plt.xlabel("max iterations")
    plt.ylabel("mean_squared_error")
    
    plt.savefig("NN.svg")

def linearRegression():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_with_more_features = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)

    print(accuracy)


if __name__ == '__main__':
   
    #decisiontreemethod() # Kaggle error of 1.27
    
    knrmethod() # Kaggle error of 2.56
    
    #randomforest()

