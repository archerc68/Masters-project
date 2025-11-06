import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import gamma, factorial

# region parameters
a, b = 0, 2*np.pi
L = b - a
x = np.linspace(a, b, 250)
t = 2 * x / L - 1
N = 25
nu = 1.9
# endregion


# region fitting a parameters functions
def model_chebyshev(x, a):
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    V = chebvander(t, N)
    return V @ a


def target_function(x):
    return np.cos(x)


def residuals(a, x):
    return model_chebyshev(x, a) - target_function(x)


a0 = np.random.random(N + 1)  # degree 5 polynomial

result = least_squares(residuals, a0, args=(x,))
optimal_a = result.x

y = model_chebyshev(x, optimal_a)


# endregion


# TODO Fix D matrix
# region D matrix
def S_matrix_vectorized(N, nu):
    i = np.arange(N + 1)[:, None]  # column vector (i)
    j = np.arange(N + 1)[None, :]  # row vector (j)
    eps_j = np.where(j == 0, 2, 1)

    # Prepare broadcasting for summation over k
    k_min = int(np.ceil(nu))
    k = np.arange(k_min, N + 1)  # vector of k values

    # Expand to broadcast: shape (i,k,j)
    I, K, J = np.meshgrid(i.squeeze(), k, j.squeeze(), indexing="ij")

    # Numerator & denominator
    sign = np.where(((I - K) % 2) == 0, 1.0, -1.0)
    num = sign * (2 * I) * factorial(I + K - 1) * gamma(K - nu + 0.5)
    den = (
        eps_j
        * (L**nu)
        * gamma(K + 0.5)
        * gamma(I - K + 1)
        * gamma(K - nu - J + 1)
        * gamma(K + J - nu + 1)
    )

    term = np.where(K <= I, num / den, 0.0)  # zero where k>i
    S = np.sum(term, axis=1)  # sum over k

    return S


def D(N, nu):
    LB = int(np.ceil(nu))

    def eps(j):
        if j == 0:
            return 2
        return 1

    D_matrix = np.zeros((N + 1, N + 1))
    for i in range(LB, N + 1):
        for k in range(LB, i + 1):
            num = (
                    (-1) ** (i - k) * 2 * i * factorial(i + k - 1) * gamma(k - nu + 0.5)
                )
            for j in range(N + 1):
                den = (
                    eps(j)
                    * (L**nu)
                    * gamma(k + 0.5)
                    * factorial(i - k)
                    * gamma(k - j - nu + 1)
                    * gamma(k + j - nu + 1)
                )
                D_matrix[i, j] += num / den
    return D_matrix

# endregion


# TODO Fix differentiation
# region finding nu derivative
D_nu = D(N=N, nu=nu)
t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
phi = chebvander(t, N)
D_nu_phi = np.linalg.tensordot(D_nu, phi, axes=(1, 1))
U = optimal_a @ D_nu_phi
# endregion


if __name__ == "__main__":
    plt.plot(x, y, label="Chebyshev fit")
    plt.plot(x, target_function(x), "--", label="Target")
    plt.plot(x, U, label=str(nu) + "th Derivative")
    plt.legend()
    plt.show()
