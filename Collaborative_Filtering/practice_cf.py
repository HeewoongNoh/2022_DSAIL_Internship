'''
<practice implementation>
implementation of latent factor collaborative filtering
There are 2 kinds of Collaborative Filtering method
1)item based collaborative filtering -nearest neighbor collaborative filtering
2)latent factor based model: Matrix factorization and its variations

Using movielens dataset for education and research, 100k size
From https://grouplens.org/datasets/movielens/

'''

# import library
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

movie_dir = 'C:/Users/heewo/anaconda3/envs/pytorch/2022_DSAIL_Intern/ml-latest-small'
#Load dataset for TruncatedSVD

rating_data = pd.read_csv(movie_dir+'/ratings.csv')
movie_data = pd.read_csv(movie_dir+'/movies.csv')

#Load dataset for svds
df_ratings = pd.read_csv(movie_dir+'/ratings.csv')
df_movies = pd.read_csv(movie_dir+'/movies.csv')


rating_data.drop('timestamp',axis=1, inplace=True)  #usedId, movieId, rating 3 columns
movie_data.drop('genres',axis=1, inplace=True) #movieId, title 2 columns

# Make rating_data and movie_data to merged one, movieId worked as a primary key
user_data_movie = pd.merge(rating_data, movie_data, on = 'movieId') # userId, movieId, rating, title 4 columns
user_data_rating = user_data_movie.pivot_table('rating',index = 'userId',columns='title').fillna(0)
# print(user_data_rating.shape) (610, 9719)
#Recommend similar movies to the particular movie

#Transpose the user_data_rating
user_data_rating_T = user_data_rating.values.T
# print(user_data_rating_T.shape) (9719, 610)

#Applying TruncatedSVD using scikit learn
SVD = TruncatedSVD(n_components=12)
matrix = SVD.fit_transform(user_data_rating_T)
# print(matrix[0]) [ 0.55545972  0.80875122 -0.370672   -0.06669671  0.63075843 -1.0096478 -0.2549328  -0.82265955 -0.77690949  0.26703128 -0.53166251  0.52380077]
corr = (np.corrcoef(matrix))
# print(corr.shape) (200,200)
movie_title = user_data_rating.columns
movie_title_list = list(movie_title)
target_movie_name = 'Avengers, The (2012)'
coff_target = movie_title_list.index(target_movie_name)
# print(movie_title_list)
corr_target_list = corr[coff_target]
# Movie recommend list (output below)
recommended_list = list(movie_title[(corr_target_list >=0.9)])[:50]
# ['Amazing Spider-Man, The (2012)', 'Annihilation (2018)', 'Ant-Man (2015)', 'Avatar (2009)', 'Avengers, The (2012)', 'Avengers: Age of Ultron (2015)', 'Big Hero 6 (2014)', 'Brave (2012)', 'Captain America: Civil War (2016)', 'Captain America: The First Avenger (2011)', 'Captain America: The Winter Soldier (2014)', 'Cloud Atlas (2012)', 'Cloudy with a Chance of Meatballs (2009)', 'Dark Knight Rises, The (2012)', 'Deadpool (2016)', 'Despicable Me (2010)', 'District 9 (2009)', 'Django Unchained (2012)', 'Doctor Strange (2016)', 'Edge of Tomorrow (2014)', "Ender's Game (2013)", 'Fantastic Beasts and Where to Find Them (2016)', 'Grand Budapest Hotel, The (2014)', 'Guardians of the Galaxy (2014)', 'Guardians of the Galaxy 2 (2017)', 'Harry Potter and the Deathly Hallows: Part 1 (2010)', 'Harry Potter and the Deathly Hallows: Part 2 (2011)', 'Hobbit: An Unexpected Journey, The (2012)', 'Hobbit: The Desolation of Smaug, The (2013)', 'How to Train Your Dragon (2010)', 'Hugo (2011)', 'Inside Out (2015)', 'Interstellar (2014)', 'Iron Man (2008)', 'Iron Man 2 (2010)', 'Iron Man 3 (2013)', 'John Wick (2014)', "King's Speech, The (2010)", 'Kingsman: The Secret Service (2015)', 'Kung Fu Panda (2008)', 'Kung Fu Panda 2 (2011)', 'Life of Pi (2012)', 'Logan (2017)', 'Looper (2012)', 'Mad Max: Fury Road (2015)', 'Man of Steel (2013)', 'Maze Runner, The (2014)', 'Now You See Me (2013)', 'Oblivion (2013)', 'Pirates of the Caribbean: On Stranger Tides (2011)']
def remove_own_rec(list):
    global target_movie_name
    for movie in recommended_list:
        if movie == target_movie_name:
            recommended_list.remove(movie)
    return recommended_list
# ['Adjustment Bureau, The (2011)', 'Amazing Spider-Man, The (2012)', 'Ant-Man (2015)', 'Avatar (2009)', 'Avengers: Age of Ultron (2015)', 'Big Hero 6 (2014)', 'Captain America: Civil War (2016)', 'Captain America: The First Avenger (2011)', 'Captain America: The Winter Soldier (2014)', 'Cloud Atlas (2012)', 'Dark Knight Rises, The (2012)', 'Deadpool (2016)', 'Despicable Me (2010)', 'District 9 (2009)', 'Django Unchained (2012)', 'Doctor Strange (2016)', 'Edge of Tomorrow (2014)', "Ender's Game (2013)", 'Fantastic Beasts and Where to Find Them (2016)', 'Girl Who Leapt Through Time, The (Toki o kakeru sh??jo) (2006)', 'Grand Budapest Hotel, The (2014)', 'Guardians of the Galaxy (2014)', 'Guardians of the Galaxy 2 (2017)', 'Harry Potter and the Deathly Hallows: Part 1 (2010)', 'Harry Potter and the Deathly Hallows: Part 2 (2011)', 'Hobbit: An Unexpected Journey, The (2012)', 'Hobbit: The Desolation of Smaug, The (2013)', 'How to Train Your Dragon (2010)', 'Hugo (2011)', 'Inside Out (2015)', 'Interstellar (2014)', 'Iron Man (2008)', 'Iron Man 2 (2010)', 'Iron Man 3 (2013)', 'John Wick (2014)', "King's Speech, The (2010)", 'Kingsman: The Secret Service (2015)', 'Kung Fu Panda (2008)', 'Kung Fu Panda 2 (2011)', 'Life of Pi (2012)', 'Logan (2017)', 'Looper (2012)', 'Mad Max: Fury Road (2015)', 'Man of Steel (2013)', 'Maze Runner, The (2014)', 'Now You See Me (2013)', 'Oblivion (2013)', 'Pirates of the Caribbean: On Stranger Tides (2011)', 'Puss in Boots (2011)']
