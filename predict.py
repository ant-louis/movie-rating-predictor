import pandas as pd 
import numpy as np
import base_methods as base

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])


df = pd.read_csv('output_list.txt', sep=" ", header=None)

users = df['user_id'].values
movies = df['movie_id'].values
ratings = []
for u,m in zip(users,movies):
    ratings.append(R[u-1][m-1])

fname = base.make_submission(ratings, df.values.squeeze(), 'MatrixFactorization')
print('Submission file "{}" successfully written'.format(fname))