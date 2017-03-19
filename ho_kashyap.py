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
#data = data - mean

# Add a 1 so that we have (d+1) * 1 vector
data = np.insert(data, 4, 1, axis=1)
#for i in range (0,2):
#  print(vec[i])
#print(vec)

#Ho-Kashyap Procedure

# Hyperplane H12
print('Hyperplane H12.')
A = np.copy(data[:10000, :])
for i in range (5000, 10000):
  A[i] = -1 * A[i]
bnplus1 = np.full((10000, 1), 1, dtype = float)
wnplus1 = np.array([ [1],[1],[1],[1] ])
prevError = 0
Aplus = np.linalg.pinv(A)

for i in range (0,200):  
  wnplus1 = np.dot(Aplus, bnplus1)
  bnplus1 = bnplus1 + (0.2) * ( (np.dot(A, wnplus1) - bnplus1) + np.fabs(np.dot(A, wnplus1) - bnplus1) )
  temp = np.dot(A, wnplus1) - bnplus1
  if( i > 1 and abs(np.linalg.norm(np.dot(temp.T, temp)) - np.linalg.norm(prevError)) <= 0.0000001 ):
    # Continuing until error is not decreasing substantially
    break
  prevError = np.dot(temp.T, temp)
  
print('After Iteration: ', i)
print(wnplus1)
print('\n')
print(bnplus1)
print('\n')
w1 = wnplus1

# Hyperplanne H23

print('Hyperplane H23.')
A = np.copy(data[5000:15000, :])

for i in range (5000, 10000):
  A[i] = -1 * A[i]
bnplus1 = np.full((10000, 1), 1, dtype = float)
wnplus1 = np.array([ [1],[1],[1],[1] ])
prevError = 0
Aplus = np.linalg.pinv(A)

for i in range (0,200):  
  wnplus1 = np.dot(Aplus, bnplus1)
  bnplus1 = bnplus1 + (0.8) * ( (np.dot(A, wnplus1) - bnplus1) + np.fabs(np.dot(A, wnplus1) - bnplus1) )
  temp = np.dot(A, wnplus1) - bnplus1
  if( i > 1 and abs(np.linalg.norm(np.dot(temp.T, temp)) - np.linalg.norm(prevError)) <= 0.0000001 ):
    # Continuing until error is not decreasing substantially
    break
  prevError = np.dot(temp.T, temp)
  
print('After Iteration: ', i)
print(wnplus1)
print('\n')
print(bnplus1)
print('\n')
w2 = wnplus1

# Hyperplanne H13

print('Hyperplane H13.')
A = np.copy(data[:5000, :])
A = np.append(A, np.copy(data[10000:15000, : ]), axis = 0)
for i in range (5000, 10000):
  A[i] = -1 * A[i]
bnplus1 = np.full((10000, 1), 1, dtype = float)
wnplus1 = np.array([ [1],[1],[1],[1] ])
prevError = 0
Aplus = np.linalg.pinv(A)

for i in range (0,200):  
  wnplus1 = np.dot(Aplus, bnplus1)
  bnplus1 = bnplus1 + (0.9) * ( (np.dot(A, wnplus1) - bnplus1) + np.fabs(np.dot(A, wnplus1) - bnplus1) )
  temp = np.dot(A, wnplus1) - bnplus1
  if( i > 1 and abs(np.linalg.norm(np.dot(temp.T, temp)) - np.linalg.norm(prevError)) <= 0.0000001 ):
    # Continuing until error is not decreasing substantially
    break
  prevError = np.dot(temp.T, temp)
  
print('After Iteration: ', i)
print(wnplus1)
print('\n')
print(bnplus1)
print('\n')
w3 = wnplus1

right = 0
wrong = 0
A = np.copy(data[:5000, :])
for i in range (0,5000):
  vec = A[i]
  a = np.dot(vec.T, w1)
  b = np.dot(vec.T, w3)
  if(a > 0 and b > 0):
    right = right + 1
  else:
    wrong = wrong + 1

print('Class 1 right, wrong = ', right, wrong)

right = 0
wrong = 0
A = np.copy(data[5000:10000, :])
for i in range (0,5000):
  vec = A[i]
  a = np.dot(vec.T, w1)
  b = np.dot(vec.T, w2)
  if(a < 0 and b > 0):
    right = right + 1
  else:
    wrong = wrong + 1

print('Class 2 right, wrong = ', right, wrong)

right = 0
wrong = 0
A = np.copy(data[10000:15000, :])
for i in range (0,5000):
  vec = A[i]
  a = np.dot(vec.T, w2)
  b = np.dot(vec.T, w3)
  if(a < 0 and b < 0):
    right = right + 1
  else:
    wrong = wrong + 1

print('Class 3 right, wrong = ', right, wrong)

# Classifying test vectors
fp = open('nvishwa_classified_hokashyap.txt', 'w')
data = np.loadtxt('test_sp2017_v19')
# Add a 1 so that we have (d+1) * 1 vector
data = np.insert(data, 4, 1, axis=1)
A = np.copy(data)
for i in range (0,15000):
  vec = A[i]
  h12 = np.dot(vec.T, w1)
  h23 = np.dot(vec.T, w2)
  h13 = np.dot(vec.T, w3)
  if(h12 > 0 and h13 > 0):
    fp.write('1')
    fp.write('\n')
  elif(h12 < 0 and h23 > 0):
    fp.write('2')
    fp.write('\n')
  elif(h23 < 0 and h13 < 0):
    fp.write('3')
    fp.write('\n')
  else:
    #print('unclassified')
    if(h12 == max(h12, h23, h13)):
      fp.write('1')
      fp.write('\n')
    elif(h23 == max(h12, h23, h13)):
      fp.write('2')
      fp.write('\n')
    elif(h13 == max(h12, h23, h13)):
      fp.write('3')
      fp.write('\n')
fp.close()
