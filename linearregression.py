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
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

#Import own script
import base_methods as base


def linearregression():
    prefix = 'Data/'


    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train = base.create_learning_matrices(R.values, user_movie_pairs)
    y_train = training_labels

    # ------------------------------- Learning ------------------------------- #
    model = LinearRegression()
    print("Training...")
    with base.measure_time('Training'):
        model.fit(X_train, y_train)

    print("Predicting")
    y_pred = model.predict(X_train)
    accuracy = mean_squared_error(training_labels, y_pred)

    print(accuracy)



if __name__ == '__main__':


    linearregression()