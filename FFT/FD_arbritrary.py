import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.special import gammaln, loggamma

plt.rcParams["font.family"] = "Times New Roman"

# ----------- Input parameters ---------- #

# Input a function f(x)


# ------------- FDE Params ------------- #

L = 4 * np.pi
N = 15


def f(x):
    return np.cos(x)


# Calculated points
j = np.arange(N + 1)
t = -np.cos(np.pi * (2 * j + 1) / (2 * (N + 1)))
x = (t + 1) * L / 2

# Interpolated points
inter = 250
t_inter = np.linspace(-1, 1, inter)
x_inter = (t_inter + 1) * L / 2



# region D matrix
def D(N, nu):
    if np.isclose(round(nu), nu):
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
    phi = chebvander(t, N).T
    phi_inter = chebvander(t_inter, N).T

    # F_T
    F_T = 2 / (N + 1) * f(x) @ phi.T
    F_T[0] *= 1 / 2

    # Plotting F^T
    plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))
    fvals = f(x_inter)
    plt.plot(x_inter, fvals, label="f(x)")
    approx = F_T @ phi_inter
    plt.plot(x_inter, approx, linestyle="--", label="F^T phi(x)")
    plt.ylabel("y")
    plt.title("Fitted F^T (N = " + str(N) + ")")
    plt.legend()
    plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))
    plt.xlabel("x")
    plt.ylabel("deviation")
    plt.plot(x_inter, approx - fvals)
    plt.plot(x_inter, np.zeros_like(x_inter), linestyle="--")
    # plt.savefig("close.png")
    plt.show()

    # Differentiation matrix
    def diff(alpha):
        D_alpha = D(N=N, nu=alpha)
        return F_T @ D_alpha @ phi_inter

    plots = 1000
    epsilon = 1e-2
    alphas = np.linspace(0 + epsilon, 2 - epsilon , plots)
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.get_cmap("plasma_r", plots)

    y_pi_two = np.empty(plots)
    for i in range(plots):
        y_i = diff(alphas[i])
        y_pi_two[i] = y_i[int(inter*np.pi/(2*L))]
        ax.plot(x_inter, y_i, color=cmap(i))

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
    plt.savefig("Cosine_FD2.png", dpi=1000)
    plt.show()

    plt.figure
    Phase_Shift = np.cos(np.pi/2 + alphas*np.pi/2)

    plt.plot(alphas, y_pi_two, label="Caputo")
    plt.plot(alphas, Phase_Shift, linestyle="--", label="Phase shift")
    plt.fill_between(alphas, y_pi_two, Phase_Shift, alpha=0.25, color="grey", hatch=".")
    plt.xlabel("Alpha")
    plt.ylabel(r"$^CD^{\alpha}\cos{(\pi/2)}$")
    plt.title("Caputo FD at " + r"$x=\pi/2$")
    plt.legend()
    plt.savefig("Phase_Shift2.png", dpi=1000)
    plt.show()



# endregion
