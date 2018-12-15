import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
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
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Build the training learning matrix
    X_train = base.create_learning_matrices(R.values, user_movie_pairs)

    # Test best parameters
    y_train = training_labels
    model = MLPRegressor(random_state = 42)
    rf_determ = RandomizedSearchCV(estimator =model, 
                                    param_distributions = grid, 
                                    cv = 2,
                                    verbose=2, 
                                    n_jobs = -1  
                                    )
    rf_determ.fit(X_train, y_train)

    print(rf_determ.best_params_)
    
    # base_model = MLPRegressor()
    # base_model.fit(X_train, y_train)
    # base_accuracy = evaluate(base_model, X_test, y_test)
    # best_determ = rf_determ.best_estimator_
    # determ_accuracy = evaluate(best_determ, X_test, y_test)

    # print('Improvement of {:0.2f}%.'.format( 100 * (determ_accuracy - base_accuracy) / base_accuracy))


def neuralnet():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    R = pd.read_csv('predicted_matrixK20.txt', sep=" ", header=None)
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Build the training learning matrix
    X_train = base.create_learning_matrices(R.values, user_movie_pairs)

    # Build the model
    y_train = training_labels
    with base.measure_time('Training'):
        print("Training...neural net")
        model = MLPRegressor(hidden_layer_sizes = (50,),
                            solver='lbfgs',
                            activation = 'logistic',
                            alpha=1e-5,
                            learning_rate_init = 0.0005,
                            learning_rate = 'constant',
                            random_state = 9,
                            early_stopping = True,
                            max_iter=10000)
        model.fit(X_train, y_train)

    # Predict
    print("Predicting...")
    y_pred_train = model.predict(X_train)

    # Checking MSE
    MSE_train = mean_squared_error(y_train, y_pred_train)
    print("MSE for mlp: {}".format(MSE_train))

    # -----------------------Submission: Running model on provided test_set---------------------------- #
    #Load test data
    test_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = base.create_learning_matrices(R.values, test_user_movie_pairs)

    # Predict
    y_pred = model.predict(X_ts)
    for i,y in enumerate(y_pred,0):
        if y_pred[i] > 5.00:
            y_pred[i] = 5.00
        if y_pred[i] < 0.00:
            y_pred[i] = 0.00

    fname = base.make_submission(y_pred, test_user_movie_pairs, 'MF_withMLP')
    print('Submission file "{}" successfully written'.format(fname))
    

if __name__ == '__main__':
    # Number of features to consider at every split
    hidden_layer_sizes = [
                          (10,),
                          (50,),
                          (100,),
                          (200,)
                        ]
                        
    activation = ['logistic','tanh','relu']
    alpha = [1e-5]
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

    #parameter_tuning(deterministic_grid)

    neuralnet()