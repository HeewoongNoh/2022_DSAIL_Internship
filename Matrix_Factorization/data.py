# import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

movie_dir = '../ml-latest-small'
random_state = 2022

def dataset_model_svd():
    #Load dataset for svds
    df_ratings = pd.read_csv(movie_dir+'/ratings.csv')
    df_movies = pd.read_csv(movie_dir+'/movies.csv')

    #data preprocessing
    df_ui_matrix = df_ratings.pivot(index= 'userId',columns='movieId',values='rating').fillna(0)
    ui_matrix = df_ui_matrix.to_numpy() # shape: (610,9724)
    ui_matrix_df = pd.DataFrame(ui_matrix,columns=df_ui_matrix.columns)
    trainset, testset = train_test_split(ui_matrix_df, test_size=0.2,random_state=random_state)

    trainset = trainset.to_numpy()
    testset = testset.to_numpy()

    # trainset = np.array(trainset, dtype=np.float64)
    # testset = np.array(testset, dtype=np.float64)

    # print(trainset.shape,testset.shape)
    return trainset, testset
# print(dataset_model_svd())


# Using csr_matrix

def dataset_model_torch():

    movie = pd.read_csv(movie_dir+'/ratings.csv')

    user_idx = {}
    for i, l in enumerate(movie['userId'].unique()):
        user_idx[l] = i
    # print(user_idx)

    movie_idx = {}
    for i, l in enumerate(movie['movieId'].unique()):
        movie_idx[l] = i
    # print(movie_idx)

    idx_user = {i: user for user, i in user_idx.items()}
    idx_movie = {i: item for item, i in movie_idx.items()}

    movie['userId'] = movie['userId'].apply(lambda x: user_idx[x]).values
    userid = movie['userId']
    movie['movieId'] = movie['movieId'].apply(lambda x: movie_idx[x]).values
    movieid = movie['movieId']
    rating = movie['rating'].values

    n_users = movie['userId'].nunique()
    n_items = movie['movieId'].nunique()

    shape = (len(set(userid)),len(set(movieid)))
    ratings = sp.csr_matrix((rating,(userid,movieid)), shape=shape)

    return ratings

# dataset_model_torch()
