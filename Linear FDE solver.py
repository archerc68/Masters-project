import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import factorial, gamma, rgamma

# ----------- Input parameters ---------- #
"Solves FDEs (Caputo) using the spectral method (Doha et al.)"

"L:             x in [0, L]"
"m:             No. Chebyshev poly. used"
"alpha:         Leading fractional derivative order"
"beta_k         RHS fractional derivative orders"
"a_k:           RHS FD coefficients"
"d_i:           d_i = y^(i)(0) -- Boundary conditions"
"g(x):          RHS perturbing function"

L = 2

# FDE Params
m = 25
alpha = 2
beta_k = np.array([1.5])
k = len(beta_k) - 1


def g(x):
    return 1 + x


a_k = np.array([-1, -1, 1])


n = int(np.floor(alpha))

d_i = np.array([1, 1])


assert alpha > np.max(beta_k)
assert len(a_k) == k + 3
assert len(d_i) == n


# region auxilliary parameters

# Auxilliary parameters
x = np.linspace(0, L, 100)
t = 2 * x / L - 1


# endregion


# region D matrix
def D_1(N):
    D_matrix_T = np.zeros((N + 1, N + 1))
    k = np.arange(1, N + 1, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4 / L
    return D_matrix


def D(N, nu):
    if type(nu) is int:
        D_matrix = np.eye(N + 1)
        D1 = D_1(N=N)
        for _ in range(nu):
            D_matrix = D1 @ D_matrix
        return D_matrix
    else:
        LB = int(np.ceil(nu))

        i = np.arange(N + 1)[:, None]
        j = np.arange(N + 1)[None, :]
        D_matrix = np.zeros((N + 1, N + 1))

        eps_j = np.ones_like(j)
        eps_j[:, 0] = 2

        for k in range(LB, N + 1):
            sign = np.where((i - k) % 2 == 0, 1, -1)
            num = sign * 2 * i * factorial(i + k - 1) * gamma(k - nu + 0.5)
            den_inv = (
                1
                / eps_j
                * 1
                / (L**nu)
                * rgamma(k + 0.5)
                * rgamma(i - k + 1)
                * rgamma(k - j - nu + 1)
                * rgamma(k + j - nu + 1)
            )
            den_inv = np.nan_to_num(den_inv, 0)
            term = np.where(k <= i, num * den_inv, 0)
            D_matrix += term
        return D_matrix


# endregion


# region Solving FDE


# Solving

# Phi(x)
phi = chebvander(t, m).T
phi_0 = phi[:, 0]


# G_T
def G_guess_var(G_0_T):
    return G_0_T @ phi - g(x)


G_0_T = np.random.random(m + 1)

result = least_squares(G_guess_var, G_0_T)

G_T = result.x

# plt.figure()
# plt.plot(x, g(x))
# plt.plot(x, G_T @ phi)
# plt.show()


# D'
D_alpha = D(N=m, nu=alpha)
D_beta_sum = np.zeros((m + 1, m + 1))
for i in range(k):
    D_beta_sum += a_k[i] * D(N=m, nu=beta_k[i])
D_prime = D_alpha - D_beta_sum - a_k[k + 1] * np.eye(m + 1)

# Operating matrix
Operator = np.empty((m + 1, m + 1))
Operator[:, :] = D_prime[:, :]

# First derivative operator
D1 = D_1(N=m)

# Boundary conditions
Operator[:, m - n + 1] = phi_0
if n >= 1:
    for i in range(1, n):
        Operator[:, m - n + 1 + i] = D1 @ Operator[:, m - n + i]

# Inverse of operator matrix
Operator_inv = np.linalg.inv(Operator)

# Column vector
column_vec = np.empty(m + 1)
column_vec[: m - n + 1] = G_T[: m - n + 1]
column_vec[m - n + 1 :] = d_i


y = column_vec @ Operator_inv @ phi


# endregion

if __name__ == "__main__":
    plt.figure()
    plt.plot(x, y, label="Tau (spectral) method")
    plt.plot(x, x + 1, linestyle="--", label="Analytical solution")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
