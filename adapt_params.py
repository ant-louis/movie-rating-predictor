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

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

def neuralNet():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_with_more_features = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    max_iters = list(range(5, 205, 5))
    for max_iter in max_iters:
        filename = "NNModel_maxiter_{}.pkl".format(max_iter)

        #Skip if the model has already been trained at this number of iterations
        if(os.path.isfile(filename)):
            print("NNModel with maxiter {} already trained. Import filename {}".format(max_iter,filename))
        else:
            start = time.time()
            with measure_time('Training'):
                model = MLPRegressor(hidden_layer_sizes = (100,), max_iter = max_iter, activation = 'logistic', random_state = 42)
                print('Training...neural net')
                model.fit(X_train, y_train)
                print("Max iterations : {} Loss : {}".format(max_iter, model.loss_))

                #Save estimator to file so that we train once
                joblib.dump(model, filename)

    models = []
    # Importing estimators from filename
    for max_iter in max_iters:
        filename = "NNModel_maxiter_{}.pkl".format(max_iter)
        print("Loading estimator {}".format(filename))
        if(os.path.isfile(filename)):
            models.append((joblib.load(filename), filename))
        else:
            break

    # Predict
    accuracies = []
    for model, filename in models:
        print("Predicting...")
        y_pred = model.predict(X_test)
        accuracy = mean_squared_error(y_test, y_pred)
        accuracies.append(accuracy)
        print("File: {} MSE: {}".format(filename,accuracy))
    
    length = len(models)
    #Plot accuracy for different max_depths
    print(accuracies)
    plt.plot(max_iters[:length],accuracies)
    plt.xlabel("max iterations")
    plt.ylabel("mean_squared_error")
    
    plt.savefig("NN.svg")

def linearRegression():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_with_more_features = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)

    print(accuracy)


if __name__ == '__main__':
   
    #decisiontreemethod() # Kaggle error of 1.27
    #randomforest()

