import numpy as np 
from mf import MF
from sklearn.decomposition import NMF


R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

mat = np.matrix(R)
with open('outfile.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')
