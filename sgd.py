# implementation of recommender system using stocastic gradient descent
# Timothy Pilien

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

print 'Users = ' + str(n_users) + ' | Movies = ' + str(n_items)

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
  R[line[1]-1, line[2]-1] = line[3]

T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
  T[line[1]-1, line[2]-1] = line[3]

# create matricies I and I2 to act as selector matricies
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# use dotproduct to predict the unknown ratings
def prediction(P, Q):
  return np.dot(P.T, Q)

# set parameters for P and Q
lmbda = 0.1
k = 20
m, n = R.shape
n_epochs = 100
gamma = 0.01

P = 3 * np.random.rand(k, m)
Q = 3 * np.random.rand(k, n)

# calculate root mean squared error
def rmse(I, R, Q, P):
  return np.sqrt(np.sum((I * (R - prediction(P, Q)))**2)/len(R[R > 0]))

train_errors = []
test_errors = []

users, items = R.nonzero()
for epoch in xrange(n_epochs):
  for u, i in zip(users, items):
    e = R[u, i] - prediction(P[:,u], Q[:,i])
    P[:,u] += gamma * (e * Q[:,i] - lmbda * P[:,u])
    Q[:,i] += gamma * (e * P[:,u] - lmbda * Q[:,i])
  train_rmse = rmse(I, R, Q, P)
  test_rmse = rmse(I2, T, Q, P)
  train_errors.append(train_rmse)
  test_errors.append(test_rmse)
  print str(epoch)

# check performance via plot
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
plt.title('SGD-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# calculate prediction matrix
R = pd.DataFrame(R)
R_hat = pd.DataFrame(prediction(P, Q))

# compare true ratings of user 22 with predictions
ratings = pd.DataFrame(data=R.loc[21, R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[21, R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
ratings 
