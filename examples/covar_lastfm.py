# Copyright 2018 Side Li and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from morpheus.algorithms.linear_regression import NormalizedLinearRegression

import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack
import morpheus.normalized_matrix as nm

s = np.matrix([])

join_set1 = np.genfromtxt('./data/LastFM/MLFK1.csv', skip_header=True, dtype=int)
r1 = mmread('./data/LastFM/MLR1Sparse.txt',)

join_set2 = np.genfromtxt('./data/LastFM/MLFK2.csv', skip_header=True, dtype=int)
r2 = mmread('./data/LastFM/MLR2Sparse.txt',)

Y = np.matrix(np.genfromtxt('./data/LastFM/MLY.csv', skip_header=True)).T
k = [join_set1 - 1, join_set2 - 1]
T = hstack((r1.tocsr()[k[0]], r2.tocsr()[k[1]]))

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
	res_fac = normalized_matrix.T * normalized_matrix
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
