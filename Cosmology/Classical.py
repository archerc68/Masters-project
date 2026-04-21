from time import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from pymittagleffler import mittag_leffler
from scipy.special import gamma as gamma_func

start = time()

# Parameters
t = np.linspace(0, 10, int(1e3))
gamma = 1.85
h = t[1] - t[0]

H_0 = 70
Omega_r0 = 9.27e-5
Omega_m0 = 0.315
Omega_lambda = 0.685


def f(t, a, a_dot):
    return (
        H_0 * a_dot * np.sqrt(Omega_r0**a ** (-4) + Omega_m0 * a ** (-3) + Omega_lambda)
    )


a = Friedmann()

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
ax1.set_ylim(1e-6, 1e2)

ax1.plot(t[indices], a[indices], color="black")


# Vertical lines

a_RM = Omega_r0 / Omega_m0
a_ML = (Omega_m0 / Omega_lambda) ** (1 / 3)

ax1.axhline(a_RM, t[0], t[-1], linewidth=1, color="k", linestyle="--")
ax1.axhline(a_ML, t[0], t[-1], linewidth=1, color="k", linestyle="--")

ax1.set_xlabel("t")
ax1.set_ylabel("a(t)")

R, M, L = Omega_r0, Omega_m0, Omega_lambda


def rad(a):
    return R / (R + a * M + a**4 * L)


def mat(a):
    return (a * M) / (R + a * M + a**4 * L)


def de(a):
    return (a**4 * L) / (R + a * M + a**4 * L)


a_dom = np.logspace(-6, 2, 250)

ax2.plot(rad(a_dom), a_dom, color="r", label="Radiation")
ax2.plot(mat(a_dom), a_dom, color="g", label="Matter")
ax2.plot(de(a_dom), a_dom, color="b", label="Dark energy")

ax2.axhline(a_RM, 0, 1, linewidth=1, color="k", linestyle="--")
ax2.axhline(a_ML, 0, 1, linewidth=1, color="k", linestyle="--")

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

# plt.savefig("figures/FracFriedNum.svg")

plt.show()
