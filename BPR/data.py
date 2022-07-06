import numpy as np
import pandas as pd

# pass in column names for each CSV

ml_dir = 'ml-100k/ml-100k'
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df = pd.read_csv(ml_dir +'/u.data', sep='\t', names=r_cols)
train_df = pd.read_csv(ml_dir+'/ua.base', sep='\t', names=r_cols)
test_df = pd.read_csv(ml_dir+'/ua.test', sep='\t', names=r_cols)

n_users = df.user_id.unique().shape[0]
n_movies = df.movie_id.unique().shape[0]

ml_100k = np.zeros((n_users, n_movies))
train = np.zeros((n_users, n_movies))
test = np.zeros((n_users, n_movies))

for row in df.itertuples():
    ml_100k[row[1] - 1, row[2] - 1] = row[3]

for row in train_df.itertuples():
    train[row[1] - 1, row[2] - 1] = row[3]

for row in test_df.itertuples():
    test[row[1] - 1, row[2] - 1] = row[3]



print(train[1].nonzero())
print("="*30)
print(train[1].nonzero()[0])
print("="*30)
x=train[1].nonzero()[0]
print(x)
import random
i = random.choice(x)
print(i)
print(np.where(x == i))
x_new = np.delete(x,np.where(x == i))
print(x_new)
