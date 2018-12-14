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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

#Import own script
import base_methods as base

#Import matrix factorization class
from sklearn.decomposition import NMF

def matrix_factorization():
    prefix='Data/'
   
    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Splitting the data set
    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs, training_labels, test_size=0.2, random_state=42)

    # Concatenating data
    user_movie_rating_triplets_training = np.hstack((X_ls, y_ls.reshape((-1, 1))))
    user_movie_rating_triplets_testing= np.hstack((X_ts, y_ts.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix_training = base.build_rating_matrix(user_movie_rating_triplets_training)
    X_train = base.create_learning_matrices(rating_matrix_training, X_ls)

    rating_matrix_testing = base.build_rating_matrix(user_movie_rating_triplets_testing)
    X_test = base.create_learning_matrices(rating_matrix_testing, X_ts)


    # Build the model
    y_train = y_ls
    y_test = y_ts
    model = NMF(n_components=2, init='random', random_state=42)

    with base.measure_time('Training'):
        print('Training...')
        W = model.fit_transform(X_train)
        H = model.components_
        print(H)

    # # ------------------------------ Prediction ------------------------------ #
    # # Load test data
    # test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # # Build the prediction matrix
    # user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))
    # rating_matrix = base.build_rating_matrix(user_movie_rating_triplets)
    # X_ts = base.create_learning_matrices(rating_matrix, test_user_movie_pairs)

    # # Predict
    # y_pred = model.predict(X_ts)

    # # Making the submission file
    # fname = make_submission(y_pred, test_user_movie_pairs, 'matrix_factorization')
    # print('Submission file "{}" successfully written'.format(fname))


if __name__ == '__main__':

    matrix_factorization()