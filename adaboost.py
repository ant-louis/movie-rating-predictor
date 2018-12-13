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

  #-------------------------------------MATRICES WITH FEATURES --------------------------------------------------------------------------
    training_user_movie_pairs_features = pd.read_csv(os.path.join(prefix,
                                                           'train_user_movie_merge.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs_features, training_labels, test_size=0.2, random_state=42)

    user_movie_rating_triplets_train = np.hstack((X_ls[['user_id','movie_id']].values,
                                            y_ls.reshape((-1, 1))))
    user_movie_rating_triplets_test = np.hstack((X_ts[['user_id','movie_id']].values,
                                            y_ts.reshape((-1, 1))))

    # Build the learning matrixtraining_with_more_features
    rating_matrix_train = base.build_rating_matrix(user_movie_rating_triplets_train)
    rating_matrix_test = base.build_rating_matrix(user_movie_rating_triplets_test)

    X_train = base.create_learning_matrices(rating_matrix_train, X_ls)
    X_test = base.create_learning_matrices(rating_matrix_test, X_ts)


    y_train = y_ls
    y_test = y_ts

    # Best estimator after hyperparameter tuning
    base_model = DecisionTreeRegressor()

    model =  AdaBoostRegressor(base_model)
    print(model)
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

    # -----------------------Submission: Running model on provided test_set---------------------------- #

    # #Load test data
    # X_test = base.load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    # X_test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # #Predict
    # print("Predicting...")
    # y_pred = model.predict(X_test)

    # fname = base.make_submission(y_pred, X_test_user_movie_pairs, 'AdaboostWithRandomForest')
    # print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    adaboost()
    # check_overfitting()