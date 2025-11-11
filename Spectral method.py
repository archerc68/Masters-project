import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import factorial, gamma, loggamma


# ----------- Input parameters ---------- #
"Takes the Caputo FD of f(x). on domain [a, b], using N Chebyshev polynomial"

"a, b:   lower and upper bounds"
"N:      No. Chebyshev poly. used"
"nu:     fractionalderivative order"
"f(x):   function used"

a, b = -1, 1
N = 10
nu = 0.6


def f(x):
    return np.exp(-x * x)


# region fitting "a" parameters to functions

# Auxilliary parameters
x = np.linspace(a, b, 250)
L = b - a
t = 2 * (x - a) / L - 1


def model_chebyshev(x, a):
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    V = chebvander(t, N)
    return a @ V.T


def residuals(a, x):
    return model_chebyshev(x, a) - f(x)


a0 = np.random.random(N + 1)  # degree 5 polynomial

result = least_squares(residuals, a0, args=(x,))
optimal_a = result.x

y = model_chebyshev(x, optimal_a)


# endregion


# TODO Fix D matrix
# region D matrix
def D_slow(N, nu):
    LB = int(np.ceil(nu))

    def eps(j):
        if j == 0:
            return 2
        return 1

    D_matrix = np.zeros((N + 1, N + 1))
    for i in range(LB, N + 1):
        for j in range(N + 1):
            for k in range(LB, i + 1):
                num = (
                    i * factorial(i + k - 1) * gamma(k - nu + 0.5)
                )
                den = (
                    eps(j)
                    
                    * gamma(k + 0.5)
                    * factorial(i - k)
                    * gamma(k - j - nu + 1)
                    * gamma(k + j - nu + 1)
                )
                scalar = (-1) ** (i - k) * 2 / (L**nu)
                D_matrix[i, j] += scalar*num / den

    return D_matrix


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

    log_eps_j = np.zeros_like(j)
    log_eps_j[:, 0] = np.log(2)

    sign = np.where((i - LB) % 2 == 0, 1, -1)

    for k in range(LB, N + 1):
        num = loggamma(i + k) + loggamma(k - nu + 0.5)
        den = (
            log_eps_j
            + loggamma(k + 0.5)
            + loggamma(i - k + 1)
            + loggamma(k - j - nu + 1)
            + loggamma(k + j - nu + 1)
        )
        frac = np.exp(-den)
        factor = i * sign * 2 * frac / (L**nu)
        sign *= -1
        term = np.where(k <= i, factor, 0)
        D_matrix += term
    return D_matrix


# endregion


# TODO Fix differentiation
# region finding nu-th FD derivative
D_nu = D_matrix_logspace(N=N, nu=nu)
phi = chebvander(t, N)  # [t, i]
D_nu_phi = D_nu @ phi.T
U = optimal_a @ D_nu_phi
# endregion


if __name__ == "__main__":
    plt.plot(x, y, label="Chebyshev fit")
    plt.plot(x, f(x), "--", label="Target")
    plt.plot(x, U, label=str(nu) + "th Derivative")
    plt.legend()
    plt.show()
