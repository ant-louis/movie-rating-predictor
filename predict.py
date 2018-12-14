import pandas as pd 
import numpy as np
import base_methods as base

df = pd.read_csv("Data/data_test.csv")
R = pd.read_csv('output_list.txt', sep=" ", header=None)

users = df['user_id'].values
movies = df['movie_id'].values
ratings = []
for u,m in zip(users,movies):
    ratings.append(R[u-1][m-1])

fname = base.make_submission(ratings, df.values.squeeze(), 'MatrixFactorization')
print('Submission file "{}" successfully written'.format(fname))