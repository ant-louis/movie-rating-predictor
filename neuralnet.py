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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

#Import own script
import base_methods as base

def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def parameter_tuning(grid):
    prefix='Data/'
    #-------------------------------------MATRIX --------------------------------------------------------------
     # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs, training_labels, test_size=0.2, random_state=42)


    user_movie_rating_triplets_training = np.hstack((X_ls,
                                            y_ls.reshape((-1, 1))))

    # Build the training learning matrix
    rating_matrix_training = base.build_rating_matrix(user_movie_rating_triplets_training)
    X_train = base.create_learning_matrices(rating_matrix_training, X_ls)

    user_movie_rating_triplets_testing= np.hstack((X_ts,
                                            y_ts.reshape((-1, 1))))

    # Build the testing learning matrix
    rating_matrix_testing = base.build_rating_matrix(user_movie_rating_triplets_testing)
    X_test = base.create_learning_matrices(rating_matrix_testing, X_ts)

    # Build the model
    y_train = y_ls
    y_test = y_ts

    # #-------------------------------------ALL FEATURES --------------------------------------------------------------------
    # training_with_more_features = base.load_from_csv(os.path.join(prefix,
    #                                                         'train_user_movie_merge.csv'))
    # training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # #----------------------------------------------------------------------------------------------------------------------
    model = MLPRegressor(random_state = 42)
    rf_determ =RandomizedSearchCV(estimator =model, 
                                    param_distributions = grid, 
                                    cv = 2,
                                    verbose=2, 
                                    n_jobs = -1  
                                    )
    rf_determ.fit(X_train, y_train)

    print(rf_determ.best_params_)
    

    base_model = MLPRegressor()
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)
    best_determ = rf_determ.best_estimator_
    determ_accuracy = evaluate(best_determ, X_test, y_test)

    print('Improvement of {:0.2f}%.'.format( 100 * (determ_accuracy - base_accuracy) / base_accuracy))




def neuralnet():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
 
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train = base.create_learning_matrices(R.values, user_movie_pairs)
    y_train = training_labels

    # #-------------------------------------ALL FEATURES --------------------------------------------------------------------
    # training_with_more_features = base.load_from_csv(os.path.join(prefix,
    #                                                         'train_user_movie_merge.csv'))
    # training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # #----------------------------------------------------------------------------------------------------------------------

    model= None
    print("Training")
    with base.measure_time('Training...neural net'):
        model = MLPRegressor(hidden_layer_sizes = (400,), 
                            activation = 'logistic', 
                            learning_rate_init = 0.0005,
                            learning_rate = 'constant',
                            early_stopping = True,
                            verbose = 2)
        model.fit(X_train, y_train)

    print(model)
    # Predict
    print("Predicting...")

    y_pred_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)

    print("Training set MSE : {}".format(MSE_train))

if __name__ == '__main__':

    # Number of features to consider at every split
    hidden_layer_sizes = [
                          (10,),
                          (50,),
                          (100,),
                          (200,)
                        ]
                        
    activation = ['logistic','tanh','relu']
    alpha = [0.0001]
    learning_rate = ['constant', 'adaptive']
    learning_rate_init = [0.0005,0.001,0.003]
    early_stopping = ['True']
    deterministic_grid = {'hidden_layer_sizes' : hidden_layer_sizes,
                        'activation' : activation,
                        'alpha':alpha,
                        'learning_rate':learning_rate,
                        'learning_rate_init':learning_rate_init,
                        'early_stopping': early_stopping
                        }

    # parameter_tuning(deterministic_grid)

    neuralnet()