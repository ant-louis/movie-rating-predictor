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
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt
@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()


def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name


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

    training_with_more_features = load_from_csv(os.path.join(prefix,
                                                            'train_user_movie_merge.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    rf_random = RandomizedSearchCV(estimator = RandomForestRegressor(), 
                                    param_distributions = grid, 
                                    n_iter = 100, 
                                    cv = 3, 
                                    verbose=2, 
                                    random_state=42, 
                                    n_jobs = -2
                                    )
    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)
    
    base_model = RandomForestRegressor(n_estimators = 10,criterion='mse', max_depth=31, random_state = 42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)

    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))




def randomforest():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    df = pd.read_csv(os.path.join(prefix, 'train_user_movie_merge.csv'), delimiter=',',dtype=float)

    train_features = df.columns
    training_with_more_features = df.values.squeeze()  

    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)


    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=31, 
                                    random_state=42,n_estimators=100, oob_score=True,n_jobs=-1)

    with measure_time('Training'):
        model.fit(X_train, y_train)
    print(model.oob_score_)
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
   
    #Select only the best 10 features
    best_features = feature_importances[:10].axes[0]
    print("Best features ... training with those again")
    df = df[best_features]
    print(best_features)
    training_with_more_features = df.values.squeeze()  


    #Train again with the reduced number of features
    X_train, X_test, y_train, y_test = train_test_split(training_with_more_features, training_labels, test_size=0.2, random_state=42)

    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=31, 
                                    random_state=42,n_estimators=100, oob_score=True,n_jobs=-1)
    
    with measure_time('Training'):
        model.fit(X_train, y_train)
        print(model.oob_score_)
    

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

    # # ----regression test overfittng python-----Submission: Running model on provided test_set---------------------------- #

    # #Load test data
    # X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    # X_test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # models = joblib.load("RandomForest_maxd_31.pkl")

    # #Predict
    # print("Predicting...")
    # y_pred = models.predict(X_test)

    # fname = make_submission(y_pred, X_test_user_movie_pairs, 'RandomForr_31')
    # print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':


    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(20, 40, num = 5)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2,4,6,8, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #             'max_features': max_features,
    #             'max_depth': max_depth,
    #             'min_samples_split': min_samples_split,
    #             'min_samples_leaf': min_samples_leaf,
    #             'bootstrap': bootstrap}

    # parameter_tuning(random_grid)

    randomforest()