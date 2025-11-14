import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import loggamma, gammaln


# ----------- Input parameters ---------- #

"Solves FDEs (Caputo) using the spectral method (Doha et al.) in the form:"


"D^alpha y(x) = sum_j [a_j D^beta_j y(x)] + a_(k+1) y(x) + a_(k+2) g(x)"

"alpha > beta_0 > beta_1 > ... > beta_k > 0"


"L:             x in [0, L]"
"m:             No. Chebyshev poly. used"
"alpha:         Leading fractional derivative order"
"beta_k         RHS fractional derivative orders"
"d_k:           RHS FD coefficients"
"a_i:           a_i = y^(a_order[i])(0) -- Boundary conditions"
"b_i:           b_i = y^(b_order[i])(L) -- Boundary conditions"
"g(x):          RHS perturbing function"


# ------------- Constants -------------- #

hbar = 6.62607015e-34 / (2 * np.pi)

# ------------- FDE Params ------------- #

L = 4*np.pi
m = 20
alpha = 1.85
beta_k = np.array([])


G = False
def g(x):
    return 0.1*np.cos(x)
    


omega = 2
d_k = np.array([-omega**2, 0])


# -------- Boundary conditions --------- #

# At x = 0
a_order = np.array([0, 1], dtype=int)
a_i = np.array([0.1, 0])


# At x = L
b_order = np.array([], dtype=int)
b_i = np.array([])


# region auxilliary parameters

n = int(np.ceil(alpha))
k = len(beta_k)

# Debug
if len(beta_k) > 0:
    assert alpha > np.max(beta_k)
assert len(d_k) == k + 2
assert len(a_i) + len(b_i) == n

# Auxilliary parameters
x = np.linspace(0, L, 250)
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
    return np.array(D_matrix)


def D(N, nu):
    if type(nu) is int:
        return np.linalg.matrix_power(D_1(N), nu)
    else:
        LB = int(np.ceil(nu))

        i = np.arange(N + 1, dtype=int)[:, None]
        j = np.arange(N + 1, dtype=int)[None, :]
        D_matrix = np.zeros((N + 1, N + 1))

        eps_j = np.ones_like(j)
        eps_j[:, 0] = 2

        coeff = 2 * i /(eps_j * L**nu)
        sign = np.where((i - LB) % 2 == 0, 1, -1)

        for k in range(LB, N + 1):
            a = k - nu + 1

            # Numerator & denominator
            log_num = loggamma(i + k) + loggamma(k - nu + 0.5)
            log_den = loggamma(k + 0.5) + gammaln(i - k + 1) + 2 * loggamma(a)

            num_den = np.exp(log_num - log_den)

            # Corrective terms to allow logarithms 
            # [loggamma(k - j - nu + 1) woud return errors]
            # Terms derived from gamma(a + j) * gamma(a - j)

            
            factors = (a + j - 1) / (a - j)
            factors[:, 0] = 1
            term = np.cumprod(factors, axis=1)

            iteration = num_den * coeff * sign/term
            sign *= -1
            
            # Masking values
            iteration = np.where(k <= i, iteration, 0)
            D_matrix += iteration

        return D_matrix
    
print(D(N=5, nu=1.5))

# endregion


# region Solving FDE


# Solving

# Phi(x)
phi = chebvander(t, m).T
phi_0 = phi[:, 0]
phi_L = phi[:, -1]


phi_BC = np.empty((n, m + 1))
phi_BC[: len(a_i), :] = phi_0
phi_BC[len(a_i) :, :] = phi_L


# G_T


if G:
    def G_guess_var(G_0_T):
        return G_0_T @ phi - g(x)


    G_0_T = np.random.random(m + 1)

    result = least_squares(G_guess_var, G_0_T)

    G_T = result.x

    # Plotting G^T
    plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))
    gvals = g(x)
    plt.plot(x, gvals, label="g(x)")
    approx = G_T @ phi
    plt.plot(x, approx, linestyle="--", label="G^T phi(x)")
    plt.ylabel("y")
    plt.title("Fitted G^T (m = " + str(m) + ")")
    plt.legend()
    plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))
    plt.xlabel("x")
    plt.ylabel("deviation")
    plt.plot(x, approx - gvals)
    plt.plot(x, np.zeros_like(x), linestyle="--")
    # plt.savefig("close.png")
    plt.show()

else:
    G_T = np.zeros(m+1)


# D'
D_alpha = D(N=m, nu=alpha)
D_beta_sum = np.zeros((m + 1, m + 1))
for i in range(k):
    D_beta_sum += d_k[i] * D(N=m, nu=beta_k[i])
D_prime = D_alpha - D_beta_sum - d_k[k] * np.eye(m + 1)

# Operating matrix
Operator = np.empty((m + 1, m + 1))
Operator[:, :] = D_prime[:, :]

# First derivative operator
D1 = D_1(N=m)

# Boundary conditions
D_order = np.concatenate((a_order, b_order), dtype=int)


for i in range(n):
    order = int(D_order[i])
    Operator[:, m - n + 1 + i] = D(N=m, nu=order) @ phi_BC[i]

# Column vector
column_vec = np.empty(m + 1)
column_vec[: m - n + 1] = G_T[: m - n + 1]
column_vec[m - n + 1 :] = np.concatenate((a_i, b_i))

C, *_ = np.linalg.lstsq(Operator.T, column_vec.T)
y = C.T @ phi


# endregion


# ---------- Plotting output ---------- #

if __name__ == "__main__":
    analytic = x * x
    plt.figure(2)  # .add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(x, y, label="Tau (spectral) method")
    # plt.plot(x, analytic, linestyle="--", label="Analytical solution")
    plt.legend()
    plt.ylabel("y")

    # plt.figure(2).add_axes((0.1, 0.1, 0.8, 0.2))
    # plt.xlabel("x")
    # plt.ylabel("deviation")
    # plt.plot(x, y - analytic)
    # plt.plot(x, np.zeros_like(x), linestyle="--")
    # # plt.savefig("y.png")
    plt.show()
