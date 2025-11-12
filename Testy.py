import numpy as np

L=2

def D_1(N):
    D_matrix_T = np.zeros((N + 1, N + 1))
    k = np.arange(1, N + 1, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4/L
    return D_matrix

D = D_1(5)
print(D)
# evals, evec = np.linalg.eig(D)
print(np.linalg.det(D))
# print(evals)
# print(evec)
