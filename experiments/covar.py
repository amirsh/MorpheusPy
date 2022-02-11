from covar_shared import * 

import os
NUM_THREADS = '1'
os.environ['OPENBLAS_NUM_THREADS'] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
import numpy as np
import time
import morpheus.normalized_matrix as nm
import pandas as pd

trials = 10
# nr = 1000000
ds = DS
# f = 5

# Cross product
print "start tesing cross product (m.T * m)"

def run_covar(f, t):
    numpy_res = []
    morpheus_res = []
    # print "Crossprod, feature ratio:", f, "tuple ratio", t
    R_df = pd.read_csv("data/R_%d_%d.csv" % (f, t), delimiter='|', header=None).to_numpy()
    S_df = pd.read_csv("data/S_%d_%d.csv" % (f, t), delimiter='|', header=None).to_numpy()
    dr = ds * f
    nr = R_df.shape[0]
    ns = S_df.shape[0]
    r = [R_df[:,1:(dr+1)]]
    s = S_df[:,1:(ds+1)]
    k = [S_df[:,0]]

    T = np.hstack((s, r[0][k[0]]))
    # T = np.hstack((s, r[0][k[0]],r[0][k[0]]))
    # print T
    # print np.dot(T.T, T)
    normalized_matrix = nm.NormalizedMatrix(s, r, k)
    # print (normalized_matrix.T * normalized_matrix)
    for _ in range(trials):
        m_start = time.time()
        tmp0 = np.dot(np.transpose(T), T)
        m_end = time.time()
        n_start = time.time()
        tmp1 = normalized_matrix.T * normalized_matrix
        n_end = time.time()
        numpy_res.append((m_end - m_start) * 1000)
        morpheus_res.append((n_end - n_start) * 1000)
    fs = str(f)
    ts = str(t)
    print ','.join([fs, ts, 'NumPy', str(np.average(numpy_res))])
    print ','.join([fs, ts, 'MorpheusPy', str(np.average(morpheus_res))])

# for t in TS:
# # for t in range(20, 21, 4):
# # for t in range(2, 3, 4):
#     run_covar(FIXED_F, t)

# for f in FS:
#     run_covar(f, FIXED_T)


for t in TS:
    for f in FS:
        run_covar(f, t)