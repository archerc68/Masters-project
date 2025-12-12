import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.special import gammaln, loggamma
from scipy.fftpack import dct

# ----------- Input parameters ---------- #

"Solves FDEs (Caputo) using the spectral method (Doha et al.) in the form:"


"D^alpha y(x) = sum_j [d_j D^beta_j y(x)] + d_(k+1) y(x) + d_(k+2) g(x)"

"alpha > beta_0 > beta_1 > ... > beta_k > 0"


"L:             x in [0, L]"
"m:             No. Chebyshev poly. used"
"alpha:         Leading fractional derivative order"
"beta_k         RHS fractional derivative orders"
"d_j:           RHS FD coefficients"
"a_i:           a_i = y^(a_order[i])(0) -- Boundary conditions"
"b_i:           b_i = y^(b_order[i])(L) -- Boundary conditions"
"g(x):          RHS perturbing function"


# ------------- FDE Params ------------- #

L = 2
m = 5
alpha = 2
beta_k = np.array([1.5])

# Argument to pass g(x)
G = 1


def g(x):
    return x**2 + 2 + 4 * np.sqrt(x / np.pi)


d_j = np.array([-1, -1, 1])


# -------- Boundary conditions --------- #

# At x = 0
a_order = np.array([0], dtype=int)
a_i = np.array([0])


# At x = L
b_order = np.array([0], dtype=int)
b_i = np.array([L**2])


# region auxilliary parameters

n = int(np.ceil(alpha))
k = len(beta_k)

# Debug
if len(beta_k) > 0:
    assert alpha > np.max(beta_k)
assert len(d_j) == k + 2
assert len(a_i) + len(b_i) == n

# Auxilliary parameters

N = 250
j = np.arange(N + 1)
t = -np.cos(np.pi * j / N)
x = (t + 1) * L / 2

# endregion


# region D matrix


def D(N, nu):
    if np.abs(round(nu) - nu) < 1e-2:

        def D_1(N):
            D_matrix_T = np.zeros((N + 1, N + 1))
            k = np.arange(1, N + 1, 2)

            for i in k:
                D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
            D_matrix = D_matrix_T.T
            D_matrix[:, 0] /= 2

            D_matrix *= 4 / L
            return np.array(D_matrix)

        return np.linalg.matrix_power(D_1(N), nu)
    else:
        LB = int(np.ceil(nu))

        i = np.arange(N + 1, dtype=int)[:, None]
        j = np.arange(N + 1, dtype=int)[None, :]
        D_matrix = np.zeros((N + 1, N + 1))

        eps_j = np.ones_like(j)
        eps_j[:, 0] = 2

        coeff = 2 * i / (eps_j * L**nu)
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

            iteration = num_den * coeff * sign / term
            sign *= -1

            # Masking values
            iteration = np.where(k <= i, iteration, 0)
            D_matrix += iteration

        return D_matrix


# endregion


# region Solving FDE


# Solving


def Solve_Collection():
    # Phi(x)
    phi = chebvander(t, m).T
    phi_0 = phi[:, 0]
    phi_L = phi[:, -1]

    phi_BC = np.empty((n, m + 1))
    phi_BC[: len(a_i), :] = phi_0
    phi_BC[len(a_i) :, :] = phi_L

    # D'
    D_alpha = D(N=m, nu=alpha)
    D_beta_sum = np.zeros((m + 1, m + 1))
    for i in range(k):
        D_beta_sum += d_j[i] * D(N=m, nu=beta_k[i])
    D_prime = D_alpha - D_beta_sum - d_j[k] * np.eye(m + 1)

    # Operating matrix
    a = np.arange((m + 1))[:, None]
    b = np.arange((m + 1))[None, :]
    i = np.arange(m + 1)
    t_cheb = -np.cos(np.pi * (2 * i + 1) / (2 * m))
    phi_cheb = chebvander(t_cheb, m).T
    x_cheb = (L / 2) * (t_cheb + 1)

    # phi_cheb = np.cos(np.pi * a * b / m)
    # x_cheb = (L / 2) * (1 - np.cos(np.pi * np.arange(m + 1) / m))

    print(D_prime.shape)
    print("PCS = " + str(phi_cheb.size))

    # Residual satisfied at interior points
    Operator = D_prime @ phi_cheb
    # Operator = dct(D_prime.T).T

    # Exterior points
    coeff_i = np.concatenate((a_i, b_i))
    coeff_order_i = np.concatenate((a_order, b_order))
    coeff_len = len(coeff_i)
    phi_i = np.empty((coeff_len, len(phi)))
    phi_i[: len(a_i)], phi_i[len(a_i) + 1 :] = phi_0, phi_L

    exterior = np.arange(coeff_len)
    exterior = np.where((exterior + 1) % 2 == 0, m - exterior + 1, exterior)

    # Column vector
    if G:
        column_vec = d_j[-1] * g(x_cheb)
    else:
        column_vec = np.zeros_like(x_cheb)

    for i in range(coeff_len):
        Operator[:, exterior[i]] = D(N=m, nu=int(coeff_order_i[i])) @ phi_i[i]
        column_vec[exterior[i]] = coeff_i[i]

    C = np.linalg.solve(Operator.T, column_vec.T)
    y = C.T @ phi
    return y


y = Solve_Collection()
# endregion


# ---------- Plotting output ---------- #

if __name__ == "__main__":
    analytic = x * x
    plt.figure(2).add_axes((0.1, 0.3, 0.8, 0.6))
    plt.plot(x, y, label="Psuedo-spectral method")
    plt.plot(x, analytic, linestyle="--", label="Analytical solution")
    plt.legend()
    plt.ylabel("y")

    plt.figure(2).add_axes((0.1, 0.1, 0.8, 0.2))
    plt.xlabel("x")
    plt.ylabel("deviation")
    plt.plot(x, y - analytic)
    plt.plot(x, np.zeros_like(x), linestyle="--")
    # # plt.savefig("y.png")
    plt.show()
