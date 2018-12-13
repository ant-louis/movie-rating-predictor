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

#Import own script
import base_method as base

def compute_cross_val(cv_val, nb_max_neighbors):
    # Loading data
    prefix = 'Data/'
    training_merge_data = base.load_from_csv(os.path.join(prefix, 'train_user_movie_merge.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Perform cross validation
    cv_results=[]
    neighbors = list(range(1,nb_max_neighbors, 5))
    for neighbor in neighbors:
        with base.measure_time('Training'):
            print('Training...with a n_neighbors of {}'.format(neighbor))
            model = KNeighborsRegressor(n_neighbors = neighbor)
            scores = cross_val_score(model, training_merge_data, training_labels, cv=cv_val, scoring='neg_mean_squared_error')
            cv_results.append(scores.mean())
    
    # Compute MSE
    MSE = [1 - x for x in cv_results]

    # Determining the best nb of nearest neighbors
    optimal_nb = neighbors[MSE.index(min(MSE))]

    # # ---------Plotting cross-validation results---------------------------- #
    # print(cv_results)
    # plt.plot(neighbors,cv_results)
    # plt.xlabel("n_neighbors")
    # plt.ylabel("Negative_mean_squared_error")
    # plt.savefig("NMSE_KNN_features_Crossval5.svg")

    return (optimal_nb, MSE)


def compute_accuracy(nb_neighbors):
    # Loading data
    prefix = 'Data/'
    training_merge_data = base.load_from_csv(os.path.join(prefix, 'train_user_movie_merge.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_merge_data, training_labels, test_size=0.1, random_state=42)

    estimator = KNeighborsRegressor(n_neighbors = nb_neighbors).fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)

    return accuracy

if __name__ == "__main__":
    # Parameters
    cv_val = 10
    nb_max_neighbors = 201

    optimal_nb, MSE = compute_cross_val(cv_val, nb_max_neighbors)
    print("The optimal number of neighbors is {}".format(optimal_nb))

    accuracy = compute_accuracy(optimal_nb)
    print("The optimal accuracy on the testing set with {} neighbors is {}".format(optimal_nb, accuracy))