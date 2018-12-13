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
    df = pd.read_csv(os.path.join(prefix, 'train_user_movie_merge.csv'), delimiter=',',dtype=float)

    train_features = df.columns
    training_with_more_features = df.values.squeeze()  

    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    # Best estimator after hyperparameter tuning
    base_model = RandomForestRegressor(bootstrap=True, n_estimators=38, criterion='mse',verbose=0)

    model =  AdaBoostRegressor(base_model)
    print(model)
    with base.measure_time('Training'):
        model.fit(X_train, y_train)
    feature_importances = pd.DataFrame(model.feature_importances_,
                                    index = train_features,
                                    columns=['importance']).sort_values('importance',ascending=False)

    print(feature_importances)
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

    fname = base.make_submission(y_pred, X_test_user_movie_pairs, 'AdaboostWithRandomForest')
    print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    adaboost()
    # check_overfitting()