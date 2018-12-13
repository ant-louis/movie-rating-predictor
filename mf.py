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
import mf from MF


def matrix_factorization():

    #-------------------------------------MATRICES --------------------------------------------------------------------------
    training_user_movie_pairs_features = base.load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs_features, training_labels, test_size=0.2, random_state=42)


    user_movie_rating_triplets_train = np.hstack((X_ls,
                                            y_ls.reshape((-1, 1))))
    user_movie_rating_triplets_test = np.hstack((X_ts,
                                            y_ts.reshape((-1, 1))))
    # Build the learning matrixtraining_with_more_features
    rating_matrix_train = base.build_rating_matrix(user_movie_rating_triplets_train)
    rating_matrix_test = base.build_rating_matrix(user_movie_rating_triplets_test)
    X_train = base.create_learning_matrices(rating_matrix_train, X_ls)
    X_test = base.create_learning_matrices(rating_matrix_test, X_ts)

    # Build the model
    y_train = y_ls
    y_test = y_ts

    mf = MF(rating_matrix_train, K=2, alpha=0.1, beta=0.01, iterations=20)
    training_process = mf.train()
    print(mf.P)
    print(mf.Q)
    print(mf.full_matrix())

if __name__ == '__main__':
    matrix_factorization()