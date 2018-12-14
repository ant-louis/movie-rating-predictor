import pandas as pd 
import numpy as np
import base_methods as base

df = pd.read_csv("Data/data_test.csv")
R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
R = R.values
users = df['user_id'].values
movies = df['movie_id'].values
ratings = []
for u,m in zip(users,movies):
    print(m)
    if (R[u-1][m-1] > 5.00) :
        ratings.append(5.00)
    else:
        ratings.append(R[u-1][m-1])




fname = base.make_submission(ratings, df.values.squeeze(), 'MatrixFactorization')
print('Submission file "{}" successfully written'.format(fname))