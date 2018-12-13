'''
Author: nvishwa@clemson.edu
'''
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()
# Ratings are 0 to 5. Biasing this list a little towards 0 because we want sparser utility matrix
ratings = [0,0,0,1,2,3,4,5]

# Test matrix for correctness checks

R_full = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
])

# Large matrix for timing and performance checks
#R_full = np.random.choice(ratings, (80, 80))
m, n = R_full.shape

# Start timer
wt = MPI.Wtime()
# Each processor gets double partitions of R_full
R_row = R_full[rank * (m // p): rank * (m // p) + (m // p), :] # A sub i
R_col = R_full[:, rank * (n // p): rank * (n // p) + (n // p)] # A sup i
#print('Processor: ', rank)
#print('R_row:', R_row)
#print('R_col:', R_col)

# Hyperparameters. Needed by all processors
k = 10
alpha = 0.001
iterations = 1000

P_row = np.random.normal(scale = 1. / k, size=(m // p, k)) # W sub i
Q_col = np.random.normal(scale = 1. / k, size=(n // p, k)) # H sup i

#print(samples)

for i in range(iterations):
  samples = [
            (i, j, R_row[i, j])
            for i in range(m // p)
            for j in range(n)
            if R_row[i, j] > 0
            ]

  np.random.shuffle(samples)
  
  # Gather Q from all processors
  Qg = np.zeros((n, k), dtype='d')
  comm.Allgather([Q_col,  MPI.DOUBLE], [Qg, MPI.DOUBLE])  
  comm.Barrier()
  for i, j, r in samples:
    prediction = P_row[i, :].dot(Qg[j, :].T)
    e = r - prediction
    P_row[i, :] += 2 * alpha * (e * Qg[j, :])

  samples = [
            (i, j, R_col[i, j])
            for i in range(m)
            for j in range(n // p)
            if R_col[i, j] > 0
            ]

  np.random.shuffle(samples)
  
  # Gather P from all processors
  Pg = np.zeros((m, k), dtype='d')
  comm.Allgather([P_row,  MPI.DOUBLE], [Pg, MPI.DOUBLE])  
  comm.Barrier()
  for i, j, r in samples:
    prediction = Pg[i, :].dot(Q_col[j, :].T)
    e = r - prediction
    Q_col[j, :] += 2 * alpha * (e * Pg[i, :])
   
comm.Barrier()

# Print results from root processor
Pg = np.zeros((m, k), dtype='d')
comm.Allgather([P_row,  MPI.DOUBLE], [Pg, MPI.DOUBLE])
Qg = np.zeros((n, k), dtype='d')
comm.Allgather([Q_col,  MPI.DOUBLE], [Qg, MPI.DOUBLE])
if rank == 0:
  #print('Input R:')
  #print(R_full)
  #print('Result R:')
  #print(np.dot(Pg, Qg.T))
  R_preds = np.dot(Pg, Qg.T)
  print('Total time with', p, ' processors:', MPI.Wtime() - wt)
  print(np.square(R_preds - R_full).mean()) 


