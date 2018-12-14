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
from mf import MF

def matrix_factorization():
    prefix='Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Concatenating data
    user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix = base.build_rating_matrix(user_movie_rating_triplets)
    sample_rating_matrix = rating_matrix[np.random.choice(rating_matrix.shape[0], 250, replace=False), :]
    sample_rating_matrix = sample_rating_matrix[:, np.random.choice(rating_matrix.shape[1], 500, replace=False)]

    # Build the model
    #for i in range(10,30):
        #print("K: %d " % (i))
    model = MF(sample_rating_matrix, K=40, alpha=0.1, beta=0.02, iterations=60)

    with base.measure_time('Training'):
        print('Training...')
        model.train()
        # print(model.P)
        # print(model.Q)
        # print(model.full_matrix())

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