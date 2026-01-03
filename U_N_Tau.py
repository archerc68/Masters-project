import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.integrate import quad

plt.rcParams["font.family"] = "Times New Roman"

n = 2
x_j = np.array([0, 0])
alpha_j = np.array([0, 1])
d_j = np.array([1, 0])
L = 10

assert len(alpha_j) == n & len(alpha_j) == len(d_j)

# ODE
gamma = 0.5
omega_0 = 2


# [0, L]
def diff_mat(alpha, N):
    D_matrix_T = np.zeros((N, N))
    k = np.arange(1, N, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4 / L
    return np.linalg.matrix_power(D_matrix, alpha)


def solve(N, x_y, Big_mat):
    Big_mat[:, -1] = (diff_mat(alpha_j[0], N) @ chebvander(-1, N - 1).T)[:, 0]
    Big_mat[:, -2] = (diff_mat(alpha_j[1], N) @ chebvander(-1, N - 1).T)[:, 0]

    Big_mat_inv = np.linalg.inv(Big_mat)

    column_vec = np.zeros(N)
    column_vec[-1], column_vec[-2] = d_j[0], d_j[1]

    a_i = column_vec @ Big_mat_inv

    x = np.linspace(0, L, 250)

    def approx(x):
        ts = 2 * x / L - 1
        return a_i @ chebvander(ts, N - 1).T
    
    def analytic(x):
        decay = np.exp(-gamma*x)
        oscillation = np.cos(omega_0*x)
        return decay*oscillation

    def approx_err(x):
        return approx(x) - analytic(x)

    mean_err = quad(approx_err, 0, 1)[0]

    if x_y:
        return x, approx(x), mean_err
    else:
        return mean_err


# U_3 plot
N = 50
ODE = diff_mat(2, N) + 2 * gamma * diff_mat(1, N) + omega_0**2 * np.eye(N)
x, y, mean_err = solve(N, x_y=True, Big_mat=ODE)
print(mean_err)
exp_decay = np.exp(-gamma * x)


plt.figure()

plt.plot(x, exp_decay, linestyle="--", color="black", label=r"$y = e^{-\gamma x}$")
plt.plot(x, -exp_decay, linestyle="--", color="black")
plt.fill_between(x, -exp_decay, exp_decay, alpha=0.5, color="blue")
plt.plot(x, y, color="red", label="Damped oscillator")

# plt.title(r"$y'' + $" + str(2 * gamma) + r"$y' + $" + str(omega_0**2) + r"$y = 0$")
plt.title(r"$\gamma = $" + str(gamma) + ", " + r"$\omega_0 = $" + str(omega_0))
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("Classical_damped.png", dpi=1000)
plt.show()
