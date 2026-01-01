import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from numpy.polynomial.chebyshev import chebvander
from scipy.integrate import quad
from scipy.optimize import curve_fit

n = 2
x_j = np.array([0, 0])
alpha_j = np.array([0, 1])
d_j = np.array([1, 0])

assert len(alpha_j) == n & len(alpha_j) == len(d_j)


# [0, 1]
def diff_mat(alpha, N):
    D_matrix_T = np.zeros((N, N))
    k = np.arange(1, N, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4
    return np.linalg.matrix_power(D_matrix, alpha)


def solve(N, x_y):
    Big_mat = diff_mat(2, N) + np.eye(N)
    Big_mat[:, -1] = (diff_mat(alpha_j[0], N) @ chebvander(-1, N - 1).T)[:, 0]
    Big_mat[:, -2] = (diff_mat(alpha_j[1], N) @ chebvander(-1, N - 1).T)[:, 0]

    Big_mat_inv = np.linalg.inv(Big_mat)

    column_vec = np.zeros(N)
    column_vec[-1], column_vec[-2] = d_j[0], d_j[1]

    a_i = column_vec @ Big_mat_inv

    x = np.linspace(0, 1, 250)

    def approx(x):
        ts = 2 * x - 1
        return a_i @ chebvander(ts, N - 1).T

    def approx_err(x):
        return approx(x) - np.cos(x)

    mean_err = quad(approx_err, 0, 1)[0]

    if x_y:
        return x, approx(x), mean_err
    else:
        return mean_err


# # U_3 plot
# x, y, mean_err = solve(5, x_y=True)


# plt.figure()
# plt.plot(x, y)
# plt.plot(x, np.cos(x))
# plt.show()


# U_N plot
plt.figure()
plt.semilogy()

Ns = np.arange(2, 30)
solves = np.vectorize(solve)
mean_err_n = solves(Ns, x_y=False)


def exp_fit(x, a, b):
    return a * np.exp(-b * x)


def exp_fit_log(x, a, b):
    return np.log(a) - b * x


mask = np.where(mean_err_n < 1e-20, 0, 1)
Ns = ma.masked_values(Ns, mask)
mean_err_n = ma.masked_values(mean_err_n, mask)

fp_err = 13

a_fit, b_fit = curve_fit(exp_fit, Ns[:fp_err], mean_err_n[:fp_err])[0]

N_cont = np.array([Ns[0], Ns[fp_err]])
label = str(round(a_fit, 2)) + r" $exp($" + str(-round(b_fit, 2)) + r"$N)$"

plt.plot(
    N_cont, exp_fit(N_cont, a_fit, b_fit), label=label, linestyle="--", color="black"
)
plt.scatter(Ns, mean_err_n, color="red")

plt.xlabel(r"$N$")
plt.ylabel("Mean Approximation Error")
plt.title(r"Error in $U_N$ approximation")
plt.legend()
plt.show()
