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

#Import own script
import base_methods as base


def adaboost():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Split data
    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs, training_labels, test_size=0.2, random_state=42)

    # Concatenate data
    user_movie_rating_triplets_training = np.hstack((X_ls, y_ls.reshape((-1, 1))))
    user_movie_rating_triplets_testing= np.hstack((X_ts, y_ts.reshape((-1, 1))))

    # Build the training learning matrix
    rating_matrix_training = base.build_rating_matrix(user_movie_rating_triplets_training)
    X_train = base.create_learning_matrices(rating_matrix_training, X_ls)

    # Build the testing learning matrix
    rating_matrix_testing = base.build_rating_matrix(user_movie_rating_triplets_testing)
    X_test = base.create_learning_matrices(rating_matrix_testing, X_ts)

    # Build the model
    y_train = y_ls
    y_test = y_ts

    # Best estimator after hyperparameter tuning
    base_model = DecisionTreeRegressor()
    model =  AdaBoostRegressor(base_model)
    with base.measure_time('Training'):
        model.fit(X_train, y_train)

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

    ## ------------------------------ Prediction ------------------------------ #
    ## Load test data
    # X_test = base.load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    # X_test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    ## Predict
    # print("Predicting...")
    # y_pred = model.predict(X_test)

    # fname = base.make_submission(y_pred, X_test_user_movie_pairs, 'AdaboostWithRandomForest')
    # print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':

    adaboost()
    