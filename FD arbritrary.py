import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import gammaln, loggamma

plt.rcParams["font.family"] = "Times New Roman"

# ----------- Input parameters ---------- #

# Input a function f(x)


# ------------- FDE Params ------------- #

L = 2 * np.pi
m = 20
alpha = 0.5


def f(x):
    return np.cos(x)


# region auxilliary parameters

n = int(np.ceil(alpha))


# Auxilliary parameters
# x = np.linspace(0, L, 250)
# t = 2 * x / L - 1

N = 250
j = np.arange(N + 1)
t = -np.cos(np.pi * j / N)
x = (t + 1) * L / 2


# endregion


# region D matrix
def D(N, nu):
    if np.abs(round(nu) - nu) < 1e-3:
        nu = int(nu)

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
if __name__ == "__main__":
    # Phi(x)
    phi = chebvander(t, m).T

    # F_T

    F_T = least_squares(lambda F_T: F_T @ phi - f(x), np.random.random(m + 1)).x

    # Plotting F^T
    plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))
    fvals = f(x)
    plt.plot(x, fvals, label="f(x)")
    approx = F_T @ phi
    plt.plot(x, approx, linestyle="--", label="F^T phi(x)")
    plt.ylabel("y")
    plt.title("Fitted F^T (m = " + str(m) + ")")
    plt.legend()
    plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))
    plt.xlabel("x")
    plt.ylabel("deviation")
    plt.plot(x, approx - fvals)
    plt.plot(x, np.zeros_like(x), linestyle="--")
    # plt.savefig("close.png")
    plt.show()

    # Differentiation matrix
    def diff(nu):
        D_alpha = D(N=m, nu=nu)
        return F_T @ D_alpha @ phi

    N = 1000
    alphas = np.linspace(0, 2, N)
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.get_cmap("plasma_r", N)
    for i in range(N):
        y_i = diff(alphas[i])
        ax.plot(x, y_i, color=cmap(i))

    # Normalizer
    norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(
        sm,
        ticks=np.linspace(alphas[0], alphas[-1], 5),
        ax=ax,
        label="Fractional derivative",
    )

    plt.title("Fractional derivatives of f(x)")
    plt.xlabel("x")
    ax.set_ylabel(r"$D^{\alpha}f(x)$")
    # plt.savefig("FD_arbritrary.png", dpi=1000)
    plt.show()


# endregion
