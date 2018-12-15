import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import base_methods as base

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
    neighbors = list(range(1,20, 1))
    for neighbor in neighbors:
        with base.measure_time('Training'):
            print('Training KNR with a n_neighbors of {}...'.format(neighbor))
            model = KNeighborsRegressor(n_neighbors = neighbor)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            MSE_train = mean_squared_error(y_train, y_pred_train)
            print("Neighbor: {} - MSE: {}".format(neighbor, MSE_train))
            MSE.append(MSE_train)
            
    index = MSE.index(min(MSE))
    optimal_nb = neighbors[index]
    print("MSE: {} - Optimal nb neighbors: {}".format(MSE[index], optimal_nb))

    # # -----------------------Submission: Running model on provided test_set---------------------------- #
    # #Load test data
    # test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # # Build the prediction matrix
    # X_ts = base.create_learning_matrices(R.values, test_user_movie_pairs)

    # # Predict
    # y_pred = model.predict(X_ts)
    # for i,y in enumerate(y_pred,0):
    #     if y_pred[i] > 5.00:
    #         y_pred[i] = 5.00
    
    # fname = base.make_submission(y_pred, test_user_movie_pairs, 'MF_withKNR')
    # print('Submission file "{}" successfully written'.format(fname))

if __name__ == "__main__":
    knr()