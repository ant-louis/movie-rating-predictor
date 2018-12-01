import pandas as pd 

train = pd.read_csv('Data/data_train.csv',dtype=object, encoding='latin-1')
user = pd.read_csv('Data/data_user.csv',dtype=object, encoding='latin-1')
movie = pd.read_csv('Data/data_movie.csv',dtype=object, encoding='latin-1')
test = pd.read_csv('Data/data_test.csv',dtype=object, encoding='latin-1')
movie.drop(columns=['movie_title','IMDb_URL',], inplace=True)
user = user.drop(['occupation'],axis=1)
pd.get_dummies(user, columns = ['gender']) #One hot encoding because Decision tree work with valuesnot strings

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

