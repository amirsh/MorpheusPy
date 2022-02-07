import numpy as np
import time
import pandas as pd
from covar_shared import * 


# ns = NS
nr = NR
# nr = 1000
# nr = 10
ds = DS
# f = 10

# Cross product
print "start generating dataset"

def generate_dataset(f, t):
    print "feature ratio:", f, "tuple ratio", t
    dr = ds * f
    # nr = ns / t
    ns = nr * t

    s = np.random.randint(100, size=(ns, ds))
    r = np.random.randint(100, size=(nr, dr))
    num = np.random.randint(nr, size=ns)
    # while (max(num) != nr - 1):
    #     num = np.random.randint(nr, size=ns)
    k = num
    # print r
    # np.savetxt("foo.csv", r, delimiter=",")
    # dt = pd.DataFrame(data=r)
    # dt.to_csv('foo.csv', mode='a', index=True)
    idx = np.arange(0, nr)[..., None]
    mat = np.hstack((idx, r))
    np.savetxt("data/R_%d_%d.csv" % (f, t), mat,"%d|" * dr + "%d")
    idx = k[..., None]
    mat = np.hstack((idx, s))
    np.savetxt("data/S_%d_%d.csv" % (f, t), mat,"%d|" * ds + "%d")

# for t in TS:
# # for t in range(20, 21, 4):
# # for t in range(2, 3, 4):
#     generate_dataset(FIXED_F, t)

# for f in FS:
#     generate_dataset(f, FIXED_T)

for t in TS:
    for f in FS:
        generate_dataset(f, t)