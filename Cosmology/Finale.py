import numpy as np
from scipy.special import gamma as gamma_func
from pymittagleffler import mittag_leffler
import matplotlib.pyplot as plt
from numba import njit


# Parameters
t = np.linspace(0, 10, int(5e5))
gamma = 1.5
h = t[1] - t[0]
const = h ** (-gamma) / gamma_func(3 - gamma)

alpha = 1
H_tilde0 = 70
Omega_r0 = 1e-4
Omega_m0 = 0.2
Omega_lambda = 0.8

a_dot_0 = 1e-4

a_eq = Omega_r0 / Omega_m0


# w terms
@njit()
def w2(alpha, k, n):
    p = 2 - alpha  # power
    if k == -1:
        return 1
    elif k == 0:
        return 2**p - 3
    elif 1 <= k <= n - 2:
        return (k + 2) ** p - 3 * (k + 1) ** p + 3 * k**p - (k - 1) ** p
    elif k == n - 1:
        return -2 * n**p + 3 * (n - 1) ** p - (n - 2) ** p
    elif k == n:
        return n**p - (n - 1) ** p


# L2 scheme with no k=-1 term
@njit()
def L2_delt(a, n, alpha=gamma):
    f_cn = 0.0
    for k in range(n):  # No k=-1 term
        f_cn += w2(alpha, k, n) * a[n - k]

    return const * f_cn


@njit()
def f(a):

    coeff = 1 / (alpha**0.5 * H_tilde0 ** (gamma - 1))

    return coeff / np.sqrt(Omega_r0 * a ** (-4) + Omega_m0 * a ** (-3) + Omega_lambda)


# Inflation

a = np.zeros_like(t)


def rad(gamma, t):
    kappa = np.sqrt(alpha * H_tilde0 ** (2 * (gamma - 1)))

    tE = t * mittag_leffler(kappa * t ** ((gamma - 1) / 2), (gamma - 1) / 2, 2)

    return np.real(a_dot_0 * tE)


for i in range(len(a)):
    a[i] = rad(gamma, t[i])
    if a[i] > 1e-4:
        nrad = i
        break


# Post-inflation
def Friedmann():
    for n in range(nrad, len(a) - 1):
        a_n = a[n]
        num = a_n + f(a_n) * L2_delt(a, n)
        den = 1 - const * f(a_n)
        a[n + 1] = num / den
    return a


a = Friedmann()

fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")
axs.set_yscale("log")

# Logarithmic Downsampling

i = np.linspace(0, np.log1p(len(a) - 1), 500)
indices = np.unique(np.expm1(i).astype(int))

# Plotting

axs.plot(t[indices], a[indices])

# Vertical lines

a_RM = Omega_r0 / Omega_m0
a_ML = (Omega_m0 / Omega_lambda) ** (1 / 3)

axs.axvline(a_RM, 0, 1, linewidth=1, color="k")
axs.axvline(a_ML, 0, 1, linewidth=1, color="k")


axs.text(
    a_RM,
    1.02,
    r"$a_{eq}^{RM}$",
    transform=axs.get_xaxis_transform(),
    ha="center",
    va="bottom",
    fontsize=10,
    clip_on=False,
)

axs.text(
    a_ML,
    1.02,
    r"$a_{eq}^{M\Lambda}$",
    transform=axs.get_xaxis_transform(),
    ha="center",
    va="bottom",
    fontsize=10,
    clip_on=False,
)

axs.set_xlabel("t")
axs.set_ylabel("a(t)")

plt.savefig("figures/FracFriedNum.svg")

plt.show()
