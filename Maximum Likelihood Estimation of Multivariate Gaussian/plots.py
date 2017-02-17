#!/usr/bin/python
# A Bayesian minimum-distance classifier, using Maximum Likelihood estimates of mean vector and covariance matrices.
# Classification is based on the minimum Mahalonobis distance, which can be considered as a discriminant function.
# Under the assumption that the data is Gaussian.
import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

f = open('train_sp2017_v19', "r")
lines = f.readlines()
f.close()

# MLE of mean for class 1

vec = []
for count in range (0,5000):
  temp = lines[count].split(' ')
  vec.append(temp)
x11 = 0
x12 = 0
x13 = 0
x14 = 0
for x in vec:
  temp = x
  x11 = x11 + float(temp[0])
  x12 = x12 + float(temp[1])
  x13 = x13 + float(temp[2])
  x14 = x14 + float(temp[3])

print("Mean for class 1:")
print(x11/5000,x12/5000,x13/5000,x14/5000)
mu1 = np.zeros(shape = (4,1))
mu1[0][0] = x11/5000
mu1[1][0] = x12/5000
mu1[2][0] = x13/5000
mu1[3][0] = x14/5000


# MLE of mean for class 2

vec = []
for count in range (5001,10000):
  temp = lines[count].split(' ')
  vec.append(temp)

x11 = 0
x12 = 0
x13 = 0
x14 = 0
for x in vec:
  temp = x
  x11 = x11 + float(temp[0])
  x12 = x12 + float(temp[1])
  x13 = x13 + float(temp[2])
  x14 = x14 + float(temp[3])

print("Mean for class 2:")
print(x11/5000,x12/5000,x13/5000,x14/5000)
mu2 = np.zeros(shape = (4,1))
mu2[0][0] = x11/5000
mu2[1][0] = x12/5000
mu2[2][0] = x13/5000
mu2[3][0] = x14/5000

# MLE of mean for class 3

vec = []
for count in range (10001, 15000):
  temp = lines[count].split(' ')
  vec.append(temp)

x11 = 0
x12 = 0
x13 = 0
x14 = 0
for x in vec:
  temp = x
  x11 = x11 + float(temp[0])
  x12 = x12 + float(temp[1])
  x13 = x13 + float(temp[2])
  x14 = x14 + float(temp[3])

print("Mean for class 3:")
print(x11/5000,x12/5000,x13/5000,x14/5000)
mu3 = np.zeros(shape = (4,1))
mu3[0][0] = x11/5000
mu3[1][0] = x12/5000
mu3[2][0] = x13/5000
mu3[3][0] = x14/5000


# MLE of covariance matrix for class 1
data = np.zeros(shape=(5000,4))
for i in range(0, 5000):
  temp = lines[i].split(' ')
  data[i][0] = float(temp[0]) - mu1[0][0]
  data[i][1] = float(temp[1]) - mu1[1][0]
  data[i][2] = float(temp[2]) - mu1[2][0]
  data[i][3] = float(temp[3]) - mu1[3][0]


covariance1 = np.dot(data.T, data)
print("Covariance class 1:")
print(covariance1 / 4999)


# MLE of covariance matrix for class 2
data = np.zeros(shape=(5000,4))
count = 0
for i in range(5001, 10000):
  temp = lines[i].split(' ')
  data[count][0] = float(temp[0]) - mu2[0][0]
  data[count][1] = float(temp[1]) - mu2[1][0]
  data[count][2] = float(temp[2]) - mu2[2][0]
  data[count][3] = float(temp[3]) - mu2[3][0]
  count = count + 1


covariance2 = np.dot(data.T, data)
print("Covariance class 2:")
print(covariance2 / 4999)


# MLE of covariance matrix for class 3
data = np.zeros(shape=(5000,4))
count = 0
for i in range(10001, 15000):
  temp = lines[i].split(' ')
  data[count][0] = float(temp[0]) - mu3[0][0]
  data[count][1] = float(temp[1]) - mu3[1][0]
  data[count][2] = float(temp[2]) - mu3[2][0]
  data[count][3] = float(temp[3]) - mu3[3][0]
  count = count + 1


covariance3 = np.dot(data.T, data)
print("Covariance class 3:")
print(covariance3 / 4999)


# HYPERPLANES

print("\nHYPERPLANE PARAMETERS wi and wi0")


print("Class 1:")
w1 = np.dot(np.linalg.inv(covariance1), mu1)
print( w1 )

w10 = 1/2.0 * np.dot( np.dot(mu1.T, np.linalg.inv(covariance1)), mu1 )
print(w10)

#---


print("Class 2:")
w2 = np.dot( np.linalg.inv(covariance2), mu2 )
print( w2 )

w20 = 1/2.0 * np.dot( np.dot(mu2.T, np.linalg.inv(covariance2)), mu2 )
print(w20)

#---

print("Class 3:")
w3 = np.dot( np.linalg.inv(covariance3), mu3 )
print( w3 )

w30 = 1/2.0 * np.dot( np.dot(mu3.T, np.linalg.inv(covariance3)), mu3 )
print(w30)


# Mahalonobis distance from Means
print("\nCLASSIFICATION OF TRAINING SET USING MINIMUM MAHALONOBIS DISTANCE")

acc = 0
x = np.zeros(shape=(4,1))
for count in range (0,5000):
  temp = lines[count].split(' ')
  x[0][0] = float(temp[0])
  x[1][0] = float(temp[1])
  x[2][0] = float(temp[2])
  x[3][0] = float(temp[3])
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(covariance1) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(covariance2) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(covariance3) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w1):
    acc = acc + 1
print "No. of elements classified in 1 from training set: ", acc

acc = 0
x = np.zeros(shape=(4,1))
for count in range (5001,10000):
  temp = lines[count].split(' ')
  x[0][0] = float(temp[0])
  x[1][0] = float(temp[1])
  x[2][0] = float(temp[2])
  x[3][0] = float(temp[3])
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(covariance1) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(covariance2) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(covariance3) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w2):
    acc = acc + 1
print "No. of elements classified in 2 from training set: ", acc

acc = 0
x = np.zeros(shape=(4,1))
for count in range (10001,15000):
  temp = lines[count].split(' ')
  x[0][0] = float(temp[0])
  x[1][0] = float(temp[1])
  x[2][0] = float(temp[2])
  x[3][0] = float(temp[3])
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(covariance1) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(covariance2) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(covariance3) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w3):
    acc = acc + 1
print "No. of elements classified in 3 from training set: ", acc

f = open('test_sp2017_v19', "r")
lines = f.readlines()
f.close()
target = open("nvishwa-classified-takehome1.txt", 'w')
x = np.zeros(shape=(4,1))
for count in range (0,15000):
  temp = lines[count].split(' ')
  x[0][0] = float(temp[0])
  x[1][0] = float(temp[1])
  x[2][0] = float(temp[2])
  x[3][0] = float(temp[3])
  maha_w1 = np.dot( np.dot( (x - mu1).T, np.linalg.inv(covariance1) ), (x - mu1))
  maha_w2 = np.dot( np.dot( (x - mu2).T, np.linalg.inv(covariance2) ), (x - mu2))
  maha_w3 = np.dot( np.dot( (x - mu3).T, np.linalg.inv(covariance3) ), (x - mu3))
  
  if( min(maha_w1, maha_w2, maha_w3) == maha_w1 ):
    target.write("1")
    target.write("\n")  
  elif( min(maha_w1, maha_w2, maha_w3) == maha_w2 ):
    target.write("2")
    target.write("\n")
  elif( min(maha_w1, maha_w2, maha_w3) == maha_w3 ):
    target.write("3")
    target.write("\n")
target.close()
