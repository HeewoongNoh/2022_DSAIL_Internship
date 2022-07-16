import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ml_dir = '../ml-latest-small'
ratings = pd.read_csv(ml_dir+"/ratings.csv") #[100836 rows x 4 columns]
# print(ratings.info())  ----each column has non null value
df_ui_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ui_matrix = df_ui_matrix.to_numpy()  # shape: (610,9724)
ui_matrix_df = pd.DataFrame(ui_matrix, columns=df_ui_matrix.columns)

# We have to preprocess the 'movies.csv' to get feature information
data = pd.read_csv(ml_dir+"/movies.csv")
all_genres = []
for x in data.genres:
    all_genres.extend(x.split("|"))

genres = pd.unique(all_genres)
zero = np.zeros((len(data),len(genres)))
dummies = pd.DataFrame(zero, columns = genres)

for i, genre in enumerate(data.genres):
    indices = dummies.columns.get_indexer(genre.split("|"))
    dummies.iloc[i,indices] = 1
data.drop(["genres"],axis=1,inplace=True)
movie_feature_df = data.join(dummies)
# movie_feature_df.to_csv("./no_index_movie_feature_encoded.csv",index=True)
movie_feature = movie_feature_df.iloc[:,2:].values.astype(float)

#If you want to split original data to train/test
# trainset, testset = train_test_split(ui_matrix_df, test_size=0.2, random_state=2023)
trainset = ui_matrix_df.to_numpy()
testset = ui_matrix_df.to_numpy()

#Include the ratings greater or equal to 4 as positive feedback
trainset = (trainset >=4).astype(float)
testset = (testset >= 4).astype(float)

#Make data binary for implicit data.
trainset = np.where(trainset>0,1,0)
testset = np.where(testset>0,1,0)


# Make negative item index list and boolean positive item list
def preprocess_dataset(trainset, testset):
    item_idx = np.arange(trainset.shape[1])
    train_neg_item = {i: item_idx[~trainset[i, :].astype(bool)] for i in range(trainset.shape[0])}
    test_neg_item = {i: item_idx[~testset[i, :].astype(bool)] for i in range(testset.shape[0])}
    train_pos_item_bool = {i: trainset[i, :].astype(bool) for i in range(trainset.shape[0])}
    test_pos_item_bool = {i: testset[i, :].astype(bool) for i in range(testset.shape[0])}

    return train_neg_item, test_neg_item, train_pos_item_bool, test_pos_item_bool






