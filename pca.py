#!/usr/bin/python
# nvishwa@clemson.edu
import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from io import StringIO

# Make all vectors zero mean
data = np.loadtxt('train_sp2017_v19')

mean = np.array([0,0,0,0])
for x in data:
  mean = mean + x

mean = (1/15000.0) * mean
data = data - mean

print(data)

# Make all vectors zero mean
test_data = np.loadtxt('test_sp2017_v19')

test_data = test_data - mean

# Add a 1 so that we have (d+1) * 1 vector
#data = np.insert(data, 4, 1, axis=1)

# PCA Procedure
sigmaX = np.cov(data.T)
print 'Covariance:'
print(sigmaX)
evals, evecs = np.linalg.eig(sigmaX)
print '\nEigen vectors'
print(evecs)
print '\nEigen values'
print(evals)

a = []
a.append(evecs[0])
a.append(evecs[2])
#a.append(evecs[1])
#a.append(evecs[3])

A = np.array(a)
print '\nA:'
print(A)

#Y = np.dot( A.T, data.T )
#print '\nY:'
#print(Y)

Y = np.dot(A, data.T)
Z = np.dot(A, test_data.T)
print '\nY:'
print(Y.T)

print '\nZ:'
print(Z.T)

# MLE based on principal components
print '\nClassification based on MLE and Mahalonobis distance based on principal components'
mu1 = (Y.T[0:5000,:]).mean(axis = 0)
mu2 = (Y.T[5000:10000,:]).mean(axis = 0)
mu3 = (Y.T[10000:15000,:]).mean(axis = 0)

print '\nMeans:'
print(mu1)
print(mu2)
print(mu3)


cov = np.cov(Y)
print '\nCovariance Matrix:'
print(cov)

acc = 0
#x = np.zeros(shape=(4,1))
for i in range (0,5000):
  x = Y.T[i]
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(cov) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(cov) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(cov) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w1):
    acc = acc + 1
print "No. of elements classified in 1 from training set: ", acc

acc = 0
#x = np.zeros(shape=(4,1))
for i in range (5000,10000):
  x = Y.T[i]
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(cov) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(cov) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(cov) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w2):
    acc = acc + 1
print "No. of elements classified in 2 from training set: ", acc

acc = 0
#x = np.zeros(shape=(4,1))
for i in range (10000,15000):
  x = Y.T[i]
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(cov) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(cov) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(cov) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w3):
    acc = acc + 1
print "No. of elements classified in 3 from training set: ", acc

# CLASSIFYING Test Data Using PCA
fp = open('nvishwa_classified_pca.txt', 'w')
for i in range (0,15000):
  x = Z.T[i]
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(cov) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(cov) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(cov) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w1):
    fp.write('1')
    fp.write('\n')
  if( min(maha_w1, maha_w2, maha_w3) == maha_w2):
    fp.write('2')
    fp.write('\n')
  if( min(maha_w1, maha_w2, maha_w3) == maha_w3):
    fp.write('3')
    fp.write('\n')
   
fp.close()
# A scatter plot of first 100 feature vectors of each class
x = Y[0]
y = Y[1]
plt.scatter(x[0:5000], y[0:5000], color = 'red')
plt.scatter(x[5000:10000], y[5000:10000], color = 'blue')
plt.scatter(x[10000:15000], y[10000:15000], color = 'green')
plt.show()


