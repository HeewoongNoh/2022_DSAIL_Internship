'''
<practice implementation>

implementation of latent factor collaborative filtering using Matrix_Factorization
There are 2 kinds of Collaborative Filtering methods
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
warnings.filterwarnings("ignore")

movie_dir = '../ml-latest-small'
#Load dataset for TruncatedSVD
rating_data = pd.read_csv(movie_dir+'/ratings.csv')
movie_data = pd.read_csv(movie_dir+'/movies.csv')

#Load dataset for svds
df_ratings = pd.read_csv(movie_dir+'/ratings.csv')
df_movies = pd.read_csv(movie_dir+'/movies.csv')

#data preprocessing
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
SVD = TruncatedSVD(n_components=15)
matrix = SVD.fit_transform(user_data_rating_T) #give (9719, 15)
corr = (np.corrcoef(matrix))
movie_title = user_data_rating.columns
movie_title_list = list(movie_title)

def give_list(movie_title_list,target_movie_name):

    coff_target = movie_title_list.index(target_movie_name)
    corr_target_list = corr[coff_target]
    recommended_list = list(movie_title[(corr_target_list >=0.9)])[:50]

    #Checking whether recommended list has input title movie
    for movie in recommended_list:
        if movie == target_movie_name:
            recommended_list.remove(movie)
    print(recommended_list)
    return recommended_list

# give_list(movie_title_list,args.title)
# ['Amazing Spider-Man, The (2012)', 'Ant-Man (2015)', 'Avatar (2009)', 'Avengers: Age of Ultron (2015)', 'Big Hero 6 (2014)', 'Captain America: The First Avenger (2011)', 'Captain America: The Winter Soldier (2014)', 'Dark Knight Rises, The (2012)', 'Deadpool (2016)', 'Despicable Me (2010)', 'District 9 (2009)', 'Edge of Tomorrow (2014)', "Ender's Game (2013)", 'Grand Budapest Hotel, The (2014)', 'Guardians of the Galaxy (2014)', 'Guardians of the Galaxy 2 (2017)', 'Harry Potter and the Deathly Hallows: Part 1 (2010)', 'Harry Potter and the Deathly Hallows: Part 2 (2011)', 'Hobbit: An Unexpected Journey, The (2012)', 'Hobbit: The Desolation of Smaug, The (2013)', 'How to Train Your Dragon (2010)', 'Inside Out (2015)', 'Interstellar (2014)', 'Iron Man (2008)', 'Iron Man 2 (2010)', 'Iron Man 3 (2013)', 'John Wick (2014)', 'Kingsman: The Secret Service (2015)', 'Kung Fu Panda (2008)', 'Life of Pi (2012)', 'Looper (2012)', 'Mad Max: Fury Road (2015)', 'Pirates of the Caribbean: On Stranger Tides (2011)', 'Rogue One: A Star Wars Story (2016)', 'Sherlock Holmes (2009)', 'Sherlock Holmes: A Game of Shadows (2011)', 'Skyfall (2012)', 'Star Trek (2009)', 'Star Trek Into Darkness (2013)', 'Star Wars: Episode VII - The Force Awakens (2015)', 'The Hunger Games (2012)', 'The Hunger Games: Mockingjay - Part 1 (2014)', 'The Lego Movie (2014)', 'Thor (2011)', 'Thor: The Dark World (2013)', 'Up (2009)', 'WALL·E (2008)', 'X-Men: Days of Future Past (2014)', 'X-Men: First Class (2011)']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-title',required=False, default='Avengers, The (2012)',help ='move title you want to search: default:Avengers, The (2012)')
    args = parser.parse_args()
    give_list(movie_title_list, args.title)