import numpy as np
import time
import pandas as pd


nr = 1000000
# nr = 1000
# nr = 10
ds = 1
f = 1

# Cross product
print "start generating dataset"

# for t in range(1, 21, 4):
for t in range(20, 21, 4):
# for t in range(2, 3, 4):
    print "feature ratio:", f, "tuple ratio", t
    dr = ds * f
    ns = nr * t

    s = np.random.randint(1000, size=(ns, ds))
    r = np.random.randint(1000, size=(nr, dr))
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
    np.savetxt("R_%d.csv" % t, mat,"%d|%d")
    idx = k[..., None]
    mat = np.hstack((idx, s))
    np.savetxt("S_%d.csv" % t, mat,"%d|%d")

