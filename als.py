# recommender system implementing ALS
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

I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

def rmse(I, R, Q, P):
  return np.sqrt(np.sum((I * (R - np.dot(P.T, Q)))**2)/len(R[R > 0]))

lmbda = 0.1
k = 20
m, n = R.shape
n_epochs = 15

P = 3 * np.random.rand(k, m)
Q = 3 * np.random.rand(k, n)
Q[0,:] = R[R != 0].mean(axis=0)
E = np.eye(k)

train_errors = []
test_errors = []

for epoch in range(n_epochs):
  for i, Ii in enumerate(I):
    nui = np.count_nonzero(Ii)
    if (nui == 0): nui = 1

    Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
    Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
    P[:,i] = np.linalg.solve(Ai, Vi)
  
  for j, Ij in enumerate(I.T):
    nmj = np.count_nonzero(Ij)
    if (nmj == 0): nmj = 1

    Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
    Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))
    Q[:,j] = np.linalg.solve(Aj, Vj)

  train_rmse = rmse(I, R, Q, P)
  test_rmse = rmse(I2, T, Q, P)
  train_errors.append(train_rmse)
  test_errors.append(test_rmse)

  print "[Epoch %d/%d] train error: %f, test error: %f" \
  %(epoch+1, n_epochs, train_rmse, test_rmse)

print "Algorithm Converged"

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data')
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data')
plt.title('ALS-WR Learning Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()

R_hat = pd.DataFrame(np.dot(P.T, Q))
R = pd.DataFrame(R)

ratings = pd.DataFrame(data=R.loc[16, R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16, R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
ratings
