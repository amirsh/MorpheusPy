from morpheus.algorithms.linear_regression import NormalizedLinearRegression
import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack

import morpheus.normalized_matrix as nm

s = mmread('./data/Expedia/MLSSparse.txt')

join_set1 = np.genfromtxt('./data/Expedia/MLFK1.csv', skip_header=True, dtype=int)
r1 = mmread('./data/Expedia/MLR1Sparse.txt')

join_set2 = np.genfromtxt('./data/Expedia/MLFK2.csv', skip_header=True, dtype=int)
r2 = mmread('./data/Expedia/MLR2Sparse.txt')

k = [join_set1 - 1, join_set2 - 1]
T = hstack((s, r1.tocsr()[k[0]], r2.tocsr()[k[1]]))
Y = np.matrix(np.genfromtxt('./data/Expedia/MLY.csv', skip_header=True, dtype=int)).T

# w_init = np.matrix(np.random.randn(T.shape[1], 1))
# gamma = 0.000001
iterations = 5
# result_eps = 1e-6

print "start factorized matrix"
normalized_matrix = nm.NormalizedMatrix(s, [r1, r2], k)
print "end factorized matrix"

m_times = []
n_times = []

import time
for x in range(iterations):
	print "start materialized covar"
	start = time.time()
	res_mat = T.T * T
	end = time.time()
	print "end materialized covar"

	m_time = end - start

	print "start factorized covar"
	start = time.time()
	res_fac = normalized_matrix.T._cross_prod()
	end = time.time()
	print "end factorized covar"

	n_time = end - start
	m_times.append(m_time)
	n_times.append(n_time)

print "times for materialized are ", m_times
print "times for factorized are ", n_times
print "average for materialized is ", np.average(m_times)
print "average for factorized is ", np.average(n_times)
print "average speedup is ", np.average(m_times) / np.average(n_times)

