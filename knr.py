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

def knr():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Build the training learning matrix
    X_train = base.create_learning_matrices(R.values, user_movie_pairs)

    # Build the model
    y_train = training_labels

    MSE = []
    neighbors = list(range(1,200, 5))
    for neighbor in neighbors:
        with base.measure_time('Training'):
            print('Training KNR with a n_neighbors of {}...'.format(neighbor))
            model = KNeighborsRegressor(n_neighbors = neighbor)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            MSE_train = mean_squared_error(y_train, y_pred_train)
            MSE.append(MSE_train)
            
    index = MSE.index(min(MSE))
    optimal_nb = neighbors[index]
    print("MSE: {} - Optimal nb neighbors: {}".format(MSE[index], optimal_nb))




    # -----------------------Submission: Running model on provided test_set---------------------------- #
    #Load test data
    test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = base.create_learning_matrices(R.values, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)
    for y in y_pred:
            if y > 5.00:
                y = 5.00
    
    fname = base.make_submission(y_pred, test_user_movie_pairs, 'MF_withKNR')
    print('Submission file "{}" successfully written'.format(fname))





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