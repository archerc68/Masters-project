import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import factorial, gamma, rgamma
import matplotlib.pyplot as plt

# ----------- Input parameters ---------- #
"Solves FDEs (Caputo) using the spectral method (Doha et al.)"

"L:             x in [0, L]"
"m:             No. Chebyshev poly. used"
"alpha:         Leading fractional derivative order"
"beta_k         RHS fractional derivative orders"        
"a_k:           RHS FD coefficients"
"d_i:           d_i = y^(i)(0) -- Boundary conditions"
"g(x):          RHS perturbing function"

L = 2 * np.pi

# FDE Params
m = 25
alpha = 3.1
beta_k = np.array([0.3, 2.1])
k = len(beta_k)
assert alpha > np.max(beta_k)


def g(x):
    return np.sin(x)


a_k = np.array([1, -1, 0.1, 1])

n = int(np.floor(alpha))

d_i = np.array([1, -1, 0])
assert len(d_i) == n


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
            1
            / eps_j
            * 1
            / (L**nu)
            * rgamma(k + 0.5)
            * rgamma(i - k + 1)
            * rgamma(k - j - nu + 1)
            * rgamma(k + j - nu + 1)
        )
        term = np.where(k <= i, num * den, 0)
        D_matrix += np.nan_to_num(term, 0)
    return np.nan_to_num(D_matrix, 0)

# endregion


# region Solving FDE



# Solving

phi = chebvander(t, m).T
phi_0 = phi[:, 0]


def G_guess_var(G_0_T):
    return G_0_T @ phi - g(x)

G_0_T = np.random.random(m + 1)

result = least_squares(G_guess_var, G_0_T)

G_T = result.x

# D'
D_alpha = D(N=m, nu=alpha)
D_beta_sum = np.zeros((m + 1, m + 1))
for i in range(k):
    D_beta_sum += a_k[i] * D(N=m, nu=beta_k[i])
D_prime = D_alpha - D_beta_sum - a_k[k + 1] * np.eye(m + 1)


Operator = np.empty((m + 1, m + 1))
Operator[:, : m - n - 1] = D_prime[:, : m - n - 1]

for i in range(n):
    Operator[:, m - n + i] = D(N=m, nu=i) @ phi_0

Operator_inv = np.linalg.inv(Operator)

column_vec = np.empty(m + 1)
column_vec[: m - n + 1] = G_T[: m - n + 1]
column_vec[m - n + 1 :] = d_i


y = (column_vec @ Operator_inv) @ phi


# endregion

if __name__ == "__main__":
    plt.figure()
    plt.plot(x, y)
    plt.show()
