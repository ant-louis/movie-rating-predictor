import pandas as pd 

train = pd.read_csv('Data/data_train.csv',dtype=object)
user = pd.read_csv('Data/data_user.csv',dtype=object)
movie = pd.read_csv('Data/data_movie_modified.csv',dtype=object)
test = pd.read_csv('Data/data_test.csv',dtype=object)

train_movie = pd.merge_ordered(train, movie, left_by="movie_id", how="outer").fillna("")
train_user_movie = pd.merge_ordered(train_movie, user, left_by="user_id", how="outer").fillna("")
train_user_movie[["user_id","movie_id"]] = train_user_movie[["user_id","movie_id"]].apply(pd.to_numeric)
train_user_movie = train_user_movie.sort_values(["user_id","movie_id"],axis=0)
train_user_movie.to_csv("Data/train_user_movie_merge.csv", index=False)


test_movie = pd.merge_ordered(test, movie, left_by="movie_id", how="outer").fillna("")
test_user_movie = pd.merge_ordered(test_movie, user, left_by="user_id", how="outer").fillna("")
test_user_movie[["user_id","movie_id"]] = test_user_movie[["user_id","movie_id"]].apply(pd.to_numeric)
test_user_movie = test_user_movie.sort_values(["user_id","movie_id"],axis=0)
test_user_movie.to_csv("Data/test_user_movie_merge.csv", index=False)

