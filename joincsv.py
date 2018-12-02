import pandas as pd 

train = pd.read_csv('Data/data_train.csv',dtype=object, encoding='latin-1')
user = pd.read_csv('Data/data_user.csv',dtype=object, encoding='latin-1')
movie = pd.read_csv('Data/data_movie.csv',dtype=object, encoding='latin-1')
test = pd.read_csv('Data/data_test.csv',dtype=object, encoding='latin-1')
train_output = pd.read_csv('Data/output_train.csv',dtype=object, encoding='latin-1')
movie.drop(columns=['movie_title','IMDb_URL','release_date','video_release_date'], axis=1, inplace=True)
user.drop(['zip_code'],axis=1, inplace=True)
user = pd.get_dummies(user, columns = ['gender']) #One hot encoding because Decision tree work with valuesnot strings
user = pd.get_dummies(user, columns = ['occupation']) #One hot encoding because Decision tree work with valuesnot strings


# wrong_zipcode = ['V3N4P','Y1A6B','E2A4H','E2A4H','L1V3W','V1G4L','T8H1N',
#                 'N4T1A','M4J2K','K7L5J','L9G2B','N2L5N','M7A1A','V5A2B','E2E3R',
#                 'R3T5K','V0R2H','V0R2M']

# index = user.index[user['zip_code'].isin(wrong_zipcode)].tolist()
# user = user[~user['zip_code'].isin(wrong_zipcode)] #Wrong zipcode
# print(index)

#Concatenating train, user and movie dataframes
train_movie = pd.merge_ordered(train, movie, left_by="movie_id", how="outer").fillna("")
train_user_movie = pd.merge_ordered(train_movie, user, left_by="user_id", how="outer").fillna("")
train_user_movie[["user_id","movie_id"]] = train_user_movie[["user_id","movie_id"]].apply(pd.to_numeric)
train_user_movie.dropna(how='any', inplace=True)
train_user_movie = train_user_movie.sort_values(["user_id","movie_id"],axis=0)


train_user_movie.to_csv("Data/train_user_movie_merge.csv", index=False)

# #Concatenating train, user and movie and output dataframes
# train_user_movie = train_user_movie.assign(rating=train_output.values)
# train_user_movie.to_csv("Data/train_user_movie_output_merge.csv", index=False)

#Concatenating test, user and movie dataframe
test_movie = pd.merge_ordered(test, movie, left_by="movie_id", how="outer").fillna("")
test_user_movie = pd.merge_ordered(test_movie, user, left_by="user_id", how="outer").fillna("")
test_user_movie[["user_id","movie_id"]] = test_user_movie[["user_id",# user.drop(['zip_code'],axis=1, inplace=True)"movie_id"]].apply(pd.to_numeric)
test_user_movie.dropna(how='any', inplace=True)
test_user_movie = test_user_movie.sort_values(["user_id","movie_id"],axis=0)

test_user_movie.to_csv("Data/test_user_movie_merge.csv", index=False)

