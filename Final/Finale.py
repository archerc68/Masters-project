from time import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from pymittagleffler import mittag_leffler
from scipy.special import gamma as gamma_func

start = time()

# Parameters
T_max = 10
N_max = int(5e5)
h = T_max / N_max

gamma = 1.85
const = h ** (-gamma) / gamma_func(3 - gamma)

alpha = 1.5
H_tilde0 = 70
Omega_r0 = 9.27e-5
Omega_m0 = 0.315
Omega_lambda = 0.685

a_dot_0 = 1e0

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


def rad(gamma, t):

    kappa = np.sqrt(alpha * H_tilde0 ** (2 * (gamma - 1)))

    tE = t * mittag_leffler(kappa * t ** ((gamma - 1) / 2), (gamma - 1) / 2, 2)

    return np.real(a_dot_0 * tE)


a = np.zeros(N_max)
for i in range(N_max):
    a[i] = rad(gamma, i * h)
    if a[i] > 1e-4:
        nrad = i
        break


# Post-inflation
def Friedmann():
    print(nrad)
    n = nrad
    while a[n] < 1e2:
        if n >= N_max - 1:
            break
        a_n = a[n]
        num = a_n + f(a_n) * L2_delt(a, n)
        den = 1 - const * f(a_n)
        a[n + 1] = num / den
        n += 1
    t = h * np.arange(n)
    return a[:n], t


a, t = Friedmann()

end = time()

print(str(end - start) + "s")

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

fig.subplots_adjust(wspace=0)
ax1.set_xscale("log")
ax1.set_yscale("log")

# Logarithmic Downsampling

i = np.linspace(0, np.log1p(len(a) - 1), 500)
indices = np.unique(np.expm1(i).astype(int))

# Plotting
ax1.set_ylim(1e-7, 1e2)
ax1.set_xlim(1e-7, 1e0)

ax1.plot(t[indices], a[indices], color="black")


# Vertical lines

a_RM = Omega_r0 / Omega_m0
a_ML = (Omega_m0 / Omega_lambda) ** (1 / 3)

ax1.axhline(a_RM, 0, 1, linewidth=1, color="k", linestyle="--")
ax1.axhline(a_ML, 0, 1, linewidth=1, color="k", linestyle="--")

ax1.set_xlabel("t")
ax1.set_ylabel("a(t)")

R, M, L = Omega_r0, Omega_m0, Omega_lambda


def rad(a):
    return R / (R + a * M + a**4 * L)


def mat(a):
    return (a * M) / (R + a * M + a**4 * L)


def de(a):
    return (a**4 * L) / (R + a * M + a**4 * L)


a_dom = np.logspace(-7, 2, 250)

ax2.plot(rad(a_dom), a_dom, color="r", label="Radiation")
ax2.plot(mat(a_dom), a_dom, color="g", label="Matter")
ax2.plot(de(a_dom), a_dom, color="b", label="Dark energy")

ax2.axhline(a_RM, 0, 1, linewidth=1, color="k", linestyle="--")
ax2.axhline(a_ML, 0, 1, linewidth=1, color="k", linestyle="--")

ax2.axvline(0.5, 1e-7, 1e2, linewidth=1, color="k", linestyle="--")

ax2.text(
    1.2,
    a_RM,
    r"$a_{eq}^{RM}$",
    transform=ax2.get_yaxis_transform(),
    ha="center",
    va="center",
    fontsize=10,
    clip_on=False,
)

ax2.text(
    1.2,
    a_ML,
    r"$a_{eq}^{M\Lambda}$",
    transform=ax2.get_yaxis_transform(),
    ha="center",
    va="center",
    fontsize=10,
    clip_on=False,
)

ax2.set_xlabel(r"$\Omega_i/\sum_j\Omega_j$")

ax2.set_xticks(np.array([0, 1 / 2, 1]))

plt.savefig("figures/FracFriedNum.svg")

plt.show()
