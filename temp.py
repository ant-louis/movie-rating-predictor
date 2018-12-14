import numpy as np 
from mf import MF

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

model = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
model.train()
print(model.P)
print(model.Q)
print(model.full_matrix())
