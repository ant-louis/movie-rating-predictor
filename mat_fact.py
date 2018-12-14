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

#Import matrix factorization class
from mf import MF

def matrix_factorization():
    prefix='Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Concatenating data
    user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix = base.build_rating_matrix(user_movie_rating_triplets)

    # Take sample of the data
    dim_user = 100
    dim_movie = 1000
    sample_rating_matrix = rating_matrix[np.random.choice(rating_matrix.shape[0], dim_user, replace=False), :]
    sample_rating_matrix = sample_rating_matrix[:, np.random.choice(rating_matrix.shape[1], dim_movie, replace=False)]

    # Build the model
    model = MF(rating_matrix, K=30, alpha=1e-5, beta=0.02, iterations=2000)
    with base.measure_time('Training'):
        print('Training...')
        model.train()
    
    # Save the predicted matrix
    predicted_matrix = np.matrix(model.full_matrix())
    with open('predicted_matrix.txt','wb') as f:
        for line in predicted_matrix:
            np.savetxt(f, line, fmt='%.5f')

    # # Compute MSE
    # predictions = []
    # true_values = []
    # test_user_index = random.sample(range(dim_user), int(dim_user / 5))
    # test_movie_index = random.sample(range(dim_movie), int(dim_movie / 5))
    # test_sample = [(i, j) for i, j in zip(test_user_index, test_movie_index) if sample_rating_matrix[i, j] != 0]
    # for i, j in test_sample:
    #     true_values.append(sample_rating_matrix[i, j])
    #     sample_rating_matrix[i, j] = 0
    #     predictions.append(predicted_matrix[i, j])
    # print("Mean squared error: ", mean_squared_error(true_values, predictions))


if __name__ == '__main__':

    matrix_factorization()