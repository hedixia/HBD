import numpy as np

seq = [2, 3, 2, 0, 1, 0, 3, 3, 3, 2, 0, 1]

act = np.asarray(np.insert(seq, [0, -1], 0) > 1, dtype=int)
act = np.diff(np.asarray(act[:-2] + act[1:-1] + act[2:] > 2.5, dtype=int))
r_start = np.where(act == 1)[0]
r_end = np.where(act == -1)[0] + 2

print(act)

#r_start = np.where(act == 1)[0]
print(r_start)

#r_end = np.where(act == -1)[0] + 2
print(r_end)
