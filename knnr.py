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
ref = list()
test = np.copy(data[12000:12001, :])
for x in data:
  ref.append(np.linalg.norm(x - test))
ref = np.array(ref)

# k = 1 Procedure
print('k = 1')
index_min = np.argmin(ref)
print(index_min)

# k = 3 Procedure
print('\nk = 3')
ref_k3 = np.copy(ref)
for i in range (0,3):
  index_min = np.argmin(ref_k3)
  print(index_min)
  ref_k3 = np.delete(ref_k3, index_min)

# k = 5 Procedure
print('\nk = 5')
ref_k5 = np.copy(ref)
for i in range (0,5):
  index_min = np.argmin(ref_k5)
  print(index_min)
  ref_k5 = np.delete(ref_k5, index_min)


# Applying k=1,3,5 for test vectors
print('\n')
test_data = np.loadtxt('test_sp2017_v19')

fp1 = open('nvishwa_classified_knnr1.txt', 'w')
fp3 = open('nvishwa_classified_knnr3.txt', 'w')
fp5 = open('nvishwa_classified_knnr5.txt', 'w')

for test in test_data:
  ref = list()
  for x in data:
    ref.append(np.linalg.norm(x - test))

  ref = np.array(ref)
  # k = 1 Procedure
  index_min = np.argmin(ref)
  if(index_min >= 0 and index_min <= 5000):
    #print 1,
    fp1.write('1')
    fp1.write('\n')
  elif(index_min >= 5001 and index_min <= 10000):
    #print 2,
    fp1.write('2')
    fp1.write('\n')
  elif(index_min >= 10001 and index_min <= 15000):
    #print 3,
    fp1.write('3')
    fp1.write('\n')

  # k = 3 Procedure
  res = dict()
  res[1] = 0
  res[2] = 0
  res[3] = 0
  ref_k3 = np.copy(ref)
  for i in range (0,3):
    index_min = np.argmin(ref_k3)
    if(index_min >= 0 and index_min <= 5000):
      res[1] = res.get(1) + 1
    elif(index_min >= 5001 and index_min <= 10000):
      res[2] = res.get(2) + 1
    elif(index_min >= 10001 and index_min <= 15000):
      res[3] = res.get(3) + 1
    ref_k3 = np.delete(ref_k3, index_min)
  
  if(res.get(1) > res.get(2) and res.get(1) > res.get(3)):
    #print 1,
    fp3.write('1')
    fp3.write('\n')
  elif(res.get(2) > res.get(1) and res.get(2) > res.get(3)):
    #print 2,
    fp3.write('2')
    fp3.write('\n')
  elif(res.get(3) > res.get(1) and res.get(3) > res.get(2)):
    #print 3,
    fp3.write('3')
    fp3.write('\n')
  #else:
  #  print('unclassified')

  # k = 5 Procedure
  res = dict()
  res[1] = 0
  res[2] = 0
  res[3] = 0
  ref_k5 = np.copy(ref)
  for i in range (0,5):
    index_min = np.argmin(ref_k3)
    if(index_min >= 0 and index_min <= 5000):
      res[1] = res.get(1) + 1
    elif(index_min >= 5001 and index_min <= 10000):
      res[2] = res.get(2) + 1
    elif(index_min >= 10001 and index_min <= 15000):
      res[3] = res.get(3) + 1
    ref_k5 = np.delete(ref_k3, index_min)
  
  if(res.get(1) > res.get(2) and res.get(1) > res.get(3)):
    #print 1
    fp5.write('1')
    fp5.write('\n')
  elif(res.get(2) > res.get(1) and res.get(2) > res.get(3)):
    #print 2
    fp5.write('2')
    fp5.write('\n')
  elif(res.get(3) > res.get(1) and res.get(3) > res.get(2)):
    #print 3
    fp5.write('3')
    fp5.write('\n')
  #else:
  #  print('unclassified')
  #break
fp1.close()
fp3.close()
fp5.close()



