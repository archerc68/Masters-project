import numpy as np
from numpy.polynomial.chebyshev import chebvander
import matplotlib.pyplot as plt
from scipy.integrate import quad

N = 20

n = 2
x_j = np.array([0, 0])
alpha_j = np.array([0, 1])
d_j = np.array([1, 0])

assert len(alpha_j) == n & len(alpha_j) == len(d_j)


# [0, 1]
def diff_mat(alpha):
    D_matrix_T = np.zeros((N, N))
    k = np.arange(1, N, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4
    return np.linalg.matrix_power(D_matrix, alpha)


Big_mat = diff_mat(2) + np.eye(N)
Big_mat[:, -1] = (diff_mat(alpha_j[0]) @ chebvander(-1, N - 1).T)[:, 0]
Big_mat[:, -2] = (diff_mat(alpha_j[1]) @ chebvander(-1, N - 1).T)[:, 0]

Big_mat_inv = np.linalg.inv(Big_mat)

column_vec = np.zeros(N)
column_vec[-1], column_vec[-2] = d_j[0], d_j[1]

a_i = column_vec @ Big_mat_inv


ts = np.linspace(-1, 1, 250)
x = (ts + 1)/2
y = a_i @ chebvander(ts, N - 1).T

mean_err = quad(lambda x: (a_i @ chebvander(x, N-1).T - np.cos(x)), 0, 1)[0]
print(mean_err)

plt.figure()
plt.plot(x, y)
plt.plot(x, np.cos(x))
plt.show()