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

import numpy as np
import time
import morpheus.normalized_matrix as nm
import plotly
import plotly.graph_objs as go
import pandas as pd

trials = 10
# nr = 1000000
ds = 1
f = 1

# Cross product
print "start tesing cross product (m.T * m)"
numpy_res = []
morpheus_res = []
# for t in range(1, 21, 4):
# for t in range(20, 21, 4):
for t in range(2, 3, 4):
    print "Crossprod, feature ratio:", f, "tuple ratio", t
    R_df = pd.read_csv("R_%d.csv" % t, delimiter='|', header=None).to_numpy()
    S_df = pd.read_csv("S_%d.csv" % t, delimiter='|', header=None).to_numpy()
    nr = R_df.shape[0]
    ns = S_df.shape[0]
    r = [R_df[:,1:2]]
    s = S_df[:,1:2]
    k = [S_df[:,0]]

    # T = np.hstack((s, r[0][k[0]]))
    # print T
    # print np.dot(T.T, T)
    normalized_matrix = nm.NormalizedMatrix(s, r, k)
    # print (normalized_matrix.T * normalized_matrix)
    for _ in range(trials):
        m_start = time.time()
        T = np.hstack((s, r[0][k[0]]))
        np.dot(T.T, T)
        # tmp0 = T.T * T
        m_end = time.time()
        n_start = time.time()
        tmp1 = normalized_matrix.T * normalized_matrix
        n_end = time.time()
        # avg.append((m_end - m_start) / (n_end - n_start))
        numpy_res.append((m_end - m_start) * 1000)
        morpheus_res.append((n_end - n_start) * 1000)

    # print (sum(avg) - min(avg) - max(avg)) / (trials - 2)
    # result.append((sum(avg) - min(avg) - max(avg)) / (trials - 2))
# total.append(result)
print numpy_res
print morpheus_res

# layout = go.Layout(
#     title='Crossprod speedup',
#     xaxis=dict(
#         title='Feature Ratio'
#     ),
#     yaxis=dict(
#         title='Tuple Ratio'
#     )
# )
# trace = go.Heatmap(z=total,
#                    x=range(1, 21),
#                    y=range(1, 5),
#                    zmin=0,
#                    zmax=6)
# data = [trace]
# fig = go.Figure(data=data, layout=layout)
# plotly.offline.plot(fig, filename='crossprod.html', show_link=False)

