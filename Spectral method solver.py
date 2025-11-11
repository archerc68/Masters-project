import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import factorial, gamma, loggamma

# ----------- Input parameters ---------- #
"Solves FDEs (Caputo) using the spectral method (Doha et al.)"

"L:             x in [0, L]"
"N:             No. Chebyshev poly. used"
"nu_lamda:      fractional derivative orders"
"a_lamda:       coeeficients in FDE"
"g(x):          function used"

L = 2 * np.pi
N = 25

# solving D^alpha f(x) = g(x)
nu_lambda = np.array([0.5, 1.2])
a_lambda = np.array([1, 0.2])
lambdas = len(a_lambda)


def g(x):
    return np.cos(x)


# region auxilliary parameters

# Auxilliary parameters
x = np.linspace(0, L, 250)
t = 2 * x / L - 1


# endregion


# region D matrix


def D(N, nu):
    LB = int(np.ceil(nu))

    i = np.arange(N + 1)[:, None]
    j = np.arange(N + 1)[None, :]
    D_matrix = np.zeros((N + 1, N + 1))

    eps_j = np.ones_like(j)
    eps_j[:, 0] = 2

    for k in range(LB, N + 1):
        sign = np.where((i - k) % 2 == 0, 1, -1)
        num = sign * 2 * i * factorial(i + k - 1) * gamma(k - nu + 0.5)
        den = (
            eps_j
            * (L**nu)
            * gamma(k + 0.5)
            * gamma(i - k + 1)
            * gamma(k - j - nu + 1)
            * gamma(k + j - nu + 1)
        )
        term = np.where(k <= i, num / den, 0)
        D_matrix += term
    return D_matrix


def D_matrix_logspace(N, nu):
    LB = int(np.ceil(nu))

    i = np.arange(N + 1)[:, None]
    j = np.arange(N + 1)[None, :]
    D_matrix = np.zeros((N + 1, N + 1))

    eps_j = np.ones_like(j)
    eps_j[:, 0] = 2
    log_eps_j = np.log(eps_j)

    for k in range(LB, N + 1):
        sign = np.where((i - k) % 2 == 0, 1, -1)
        num = np.log(i) + loggamma(i + k) + loggamma(k - nu + 0.5)
        den = (
            log_eps_j
            + loggamma(k + 0.5)
            + loggamma(i - k + 1)
            + loggamma(k + j - nu + 1)
        )
        frac = np.exp(num - den)
        term = np.where(k <= i, sign * 2 * frac / (gamma(k - j - nu + 1) * L**nu), 0)
        D_matrix += term
    return D_matrix


# endregion


# region Solving FDE


phi = chebvander(t, N).T

D_sum = np.zeros((N + 1, N + 1))
for i in range(lambdas):
    D_sum += a_lambda[i] * D_matrix_logspace(N=N, nu=nu_lambda[i])
D_sum_phi = D_sum @ phi

C_T0 = np.random.random(N + 1)


def var(C_T0, x):
    return C_T0 @ D_sum_phi - g(x)


result = least_squares(var, C_T0, args=(x,))

C_T = result.x

y = C_T @ phi

# endregion

if __name__ == "__main__":
    plt.plot(x, y)
    plt.show()
