import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import gamma

a, b = 0, 10
L = b - a
x = np.linspace(a, b, 250)
t = 2 * x / L - 1
N = 10


def S_matrix_vectorized(N, nu):
    i = np.arange(N)[:, None]  # column vector (i)
    j = np.arange(N)[None, :]  # row vector (j)
    eps_j = np.where(j == 0, 2, 1)

    # Prepare broadcasting for summation over k
    k_min = int(np.ceil(nu))
    k = np.arange(k_min, N)  # vector of k values

    # Expand to broadcast: shape (i,k,j)
    I, K, J = np.meshgrid(i.squeeze(), k, j.squeeze(), indexing="ij")

    # Numerator & denominator
    sign = np.where(((I - K) % 2) == 0, 1.0, -1.0)
    num = sign * (2**I) * gamma(I + K) * gamma(K - nu + 0.5)
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


D_nu = S_matrix_vectorized(N=N, nu=1.5)
phi = chebvander(t, N - 1)
D_nu_phi = np.linalg.tensordot(D_nu, phi, axes=(1, 1))


def model_chebyshev(x, a):
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    V = chebvander(t, N - 1)
    return V @ a


def target_function(x):
    return np.sin(x)


def residuals(a, x):
    return model_chebyshev(x, a) - target_function(x)


if __name__ == "__main__":
    a0 = np.random.random(N)  # degree 5 polynomial

    result = least_squares(residuals, a0, args=(x,))
    optimal_a = result.x

    y = model_chebyshev(x, optimal_a)
    plt.plot(x, y, label="Chebyshev fit")
    plt.plot(x, target_function(x), "--", label="Target")
    plt.legend()
    plt.show()

    U = np.linalg.tensordot(optimal_a, D_nu_phi, axes=(0, 0))

    plt.figure()
    plt.plot(x, U)
    plt.show()
