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
model = NMF( init='random', random_state=42)
H = model.fit_transform(R)
W = model.components_
nR = np.dot(H,W)
print(nR)




# model = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
# model.train()
# print(model.P)
# print(model.Q)
# print(model.full_matrix())
