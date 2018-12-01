import pandas as pd 

train = pd.read_csv('Data/data_train.csv',dtype=object)
user = pd.read_csv('Data/data_user.csv',dtype=object)
movie = pd.read_csv('Data/data_movie_modified.csv',dtype=object)
test = pd.read_csv('Data/data_test.csv',dtype=object)

train_user_merge = train.merge(user, on="user_id", how="outer").fillna("")
train_user_movie_merge = train_user_merge.merge(movie, on="movie_id", how="outer").fillna("")
train_user_movie_merge.to_csv("Data/train_user_movie_merge.csv", index=False)

test_user_merge = test.merge(user, on="user_id", how="outer").fillna("")
test_user_movie_merge = test_user_merge.merge(movie, on="movie_id", how="outer").fillna("")
test_user_movie_merge.to_csv("Data/test_user_movie_merge.csv", index=False)

