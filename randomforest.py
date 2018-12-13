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
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_ls, X_ts, y_ls, y_ts = train_test_split(training_user_movie_pairs, training_labels, test_size=0.2, random_state=42)


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


def check_overfitting():
    prefix = 'Data/'
    df = pd.read_csv(os.path.join(prefix, 'train_user_movie_merge.csv'), delimiter=',',dtype=float)

    train_features = df[['user_id','movie_id','age']].columns
    training_with_more_features = df[['user_id','movie_id','age']].values.squeeze()     

    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)


    train_MSE= []
    test_MSE= []
    n_est = list(range(1,100, 1))
    for n in n_est:
        with base.measure_time('Training'):
            print('Training...with a n-est of {}'.format(n))
            model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=31, 
                                            random_state=42,n_estimators=n, oob_score=True,n_jobs=-1)
            model.fit(X_train, y_train)

            #Check for overfitting
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            test_MSE.append(mean_squared_error(y_test, y_pred_test))
            train_MSE.append(mean_squared_error(y_train, y_pred_train))
            feature_importances = pd.DataFrame(model.feature_importances_,
                        index = train_features,
                        columns=['importance']).sort_values('importance',ascending=False)
            
            print(feature_importances[:10])


    print("Test set MSE: {}".format(test_MSE))
    print("Train set MSE: {}".format(train_MSE))


    # plt.xlabel("maxdepths")
    # plt.ylabel("mean_squared_error")

    plt.plot(n_est, train_MSE, label='Train')
    plt.plot(n_est, test_MSE, label='Test')
    plt.legend(loc='lower left')
    plt.xlabel('n_est')
    plt.ylabel('MSE')
    plt.savefig('NumberEstMSE',format='svg')


    
def randomforest():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    df = pd.read_csv(os.path.join(prefix, 'train_user_movie_merge.csv'), delimiter=',',dtype=float)

    train_features = df.columns
    training_with_more_features = df.values.squeeze()  

    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # Best estimator after hyperparameter tuning
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=11, min_samples_leaf= 8, min_samples_split = 5,
                                    random_state=42,n_estimators=38, oob_score=True,n_jobs=-1)

    print(model)
    with base.measure_time('Training'):
        model.fit(X_train, y_train)
    feature_importances = pd.DataFrame(model.feature_importances_,
                                    index = train_features,
                                    columns=['importance']).sort_values('importance',ascending=False)

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
    X_test = base.load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    X_test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    #Predict
    print("Predicting...")
    y_pred = model.predict(X_test)

    fname = base.make_submission(y_pred, X_test_user_movie_pairs, 'RandomForr_tuned')
    print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    # Number of features to consider at every split
    n_estimators = list(range(36,44,2))
    max_depth= list(range(9,12,1))
    # Minimum number of samples required to split a node
    min_samples_split = [5,6,7]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4,8,10]
    # Create the random grid
    deterministic_grid = {'n_estimators' : n_estimators,
                        'max_depth' : max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                        }

    # parameter_tuning(deterministic_grid)

    randomforest()
    # check_overfitting()