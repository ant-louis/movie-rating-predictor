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

def parameter_tuning(grid):
    prefix='Data/'


    #-------------------------------------MATRICES --------------------------------------------------------------------------
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

    # Build the model
    y_train = y_ls
    y_test = y_ts

    # #-------------------------------------ALL FEATURES --------------------------------------------------------------------
    # training_with_more_features = base.load_from_csv(os.path.join(prefix,
    #                                                         'train_user_movie_merge.csv'))
    # training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # #----------------------------------------------------------------------------------------------------------------------
    model =  RandomForestRegressor( criterion='mse', max_features = 'auto',bootstrap=True, random_state = 42)
    rf_determ = GridSearchCV(estimator =model, 
                                    param_grid = grid, 
                                    cv = 2,
                                    verbose=2, 
                                    n_jobs = 4  
                                    )
    rf_determ.fit(X_train, y_train)

    print(rf_determ.best_params_)
    
    base_model = RandomForestRegressor(criterion='mse',max_features = 'auto',bootstrap=True, determ_state = 42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)
    best_determ = rf_determ.best_estimator_
    determ_accuracy = evaluate(best_determ, X_test, y_test)

    print('Improvement of {:0.2f}%.'.format( 100 * (determ_accuracy - base_accuracy) / base_accuracy))

def randomforest():
    prefix = 'Data/'

  #-------------------------------------MATRICES WITH FEATURES --------------------------------------------------------------------------


    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train = base.create_learning_matrices(R.values, user_movie_pairs)
    y_train = training_labels

    # ------------------------------- Learning ------------------------------- #

    # #-------------------------------------ALL FEATURES --------------------------------------------------------------------
    # training_with_more_features = base.load_from_csv(os.path.join(prefix,
    #                                                         'train_user_movie_merge.csv'))
    # training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # #----------------------------------------------------------------------------------------------------------------------

    # Best estimator after hyperparameter tuning
    model = RandomForestRegressor(bootstrap=True, 
                                    criterion='mse',
                                    random_state=42,
                                    n_estimators=38,
                                    n_jobs=-1,
                                    verbose = 2,
                                    max_depth= 9)

    print(model)
    print("Training...")
    with base.measure_time('Training'):
        model.fit(X_train, y_train)

    #Check for overfitting
    y_pred_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    print("Train set MSE: {}".format(MSE_train))

    # #Plot accuracy for different max_depths
    # print(accuracies)
    # plt.plot(maxdepths,accuracies)
    # plt.xlabel("maxdepths")
    # plt.ylabel("mean_squared_error")
    # plt.savefig("RandomForest_precise.svg")

    # -----------------------Submission: Running model on provided test_set---------------------------- #

    #Load test data
    # X_test = base.load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    #Predict
    print("Predicting...")
    # Build the prediction matrix
    X_ts = base.create_learning_matrices(R.values, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)

    for i,y in enumerate(y_pred,0):
        if y_pred[i] > 5.00:
            y_pred[i] = 5.00
    
    fname = base.make_submission(y_pred, test_user_movie_pairs, 'MF_withRandomForest')
    print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    # # Number of features to consider at every split
    # n_estimators = list(range(36,44,2))
    # max_depth= list(range(9,12,1))
    # # Minimum number of samples required to split a node
    # min_samples_split = [5,6,7]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [4,8,10]
    # # Create the random grid
    # deterministic_grid = {'n_estimators' : n_estimators,
    #                     'max_depth' : max_depth,
    #                     'min_samples_split': min_samples_split,
    #                     'min_samples_leaf': min_samples_leaf
    #                     }

    # parameter_tuning(deterministic_grid)

    randomforest()
    # check_overfitting()