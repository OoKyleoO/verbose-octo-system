# Implementation of Recommender System for use with MovieLens Database
# Timothy Pilien

import numpy as np
import pandas as pd

# read in the u.data file which contains the full dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# count the number of unique users and movies
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print '# Users = ' + str(n_users) + ' | # Movies ' + str(n_items)

# split the dataset into testing and training
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

# create two matrices (user-item) one is for training the other for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
  train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
  test_data_matrix[line[1]-1, line[2]-1] = line[3]

# calculate the cosine similarity
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine') 

# making predictions
def predict(ratings, similarity, type='user'):
  if type == 'user':
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
  elif type == 'item':
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
  return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

# use root mean squared error to evaluate accuracy
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
  prediction = prediction[ground_truth.nonzero()].flatten()
  ground_truth = ground_truth[ground_truth.nonzero()].flatten()
  return sqrt(mean_squared_error(prediction, ground_truth))

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))

# calculate sparisty of MovieLens dataset
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print 'Sparsity Level of MovieLens100k: ' + str(sparsity*100) + '%'

# use SVD for model based content filtering
import scipy.sparse as sp
from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF RMSE (Model): ' + str(rmse(X_pred, test_data_matrix))
