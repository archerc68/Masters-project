import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

R, M, L = 9.27e-5, 0.315, 0.685

def rad(a):
    return R/(R + a*M + a**4 * L)

def mat(a):
    return (a*M)/(R + a*M + a**4 * L)

def de(a):
    return (a**4 * L)/(R + a*M + a**4 * L)

a = np.logspace(-7, 2, 250)

fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")

plt.axvline(R/M, 0, 1, linewidth=1, color = 'k')
plt.axvline((M/L)**(1/3), 0, 1, linewidth=1, color = 'k')

plt.axhline(0, a[0], a[-1], linewidth=1, color = 'k')
plt.axhline(1, a[0], a[-1], linewidth=1, color = 'k')

axs.plot(a, rad(a), label="Radiation", color="r")
axs.plot(a, mat(a), label="Matter", color="g")
axs.plot(a, de(a), label="Dark Energy", color="b")

axs.set_xlabel(r"$a$")
axs.set_ylabel(r"$\Omega_i/\sum_j\Omega_j$")

axs.annotate(r"$a_{eq}^{RM}=\Omega_{R0}/\Omega_{M0}$",
             xy = (R/M, 0.5),
             xytext= (3*R/M, 0.5),
             arrowprops=dict(facecolor='black', arrowstyle="->"))

axs.annotate(r"$a_{eq}^{M\Lambda}=\sqrt[3]{\Omega_{M0}/\Omega_{\Lambda0}}$",
             xy = ((M/L)**(1/3), 0.5),
             xytext= (3*(M/L)**(1/3), 0.5),
             arrowprops=dict(facecolor='black', arrowstyle="->"))

axs.legend()


plt.savefig("figures/Domination.svg")

plt.show()