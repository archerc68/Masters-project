import numpy as np
import matplotlib.pyplot as plt
from differint.differint import PCsolver
from fodeint import caputoEuler
import matplotlib as mpl


def fractional_diff_eq(a, t):
    return a * np.sqrt(0.2 * a ** (-3) + 1e-4 * a ** (-4) + 0.8)


def classical_diff_eq(a, t):
    return a * np.sqrt(0.2 * a ** (-3) + 1e-4 * a ** (-4) + 0.8)


a0, a1 = 1e-4, 100
t = np.linspace(a0, a1, 250)

# a_m = caputoEuler(0.1, fractional_diff_eq, 1e-4, t)
# a_0 = caputoEuler(0.999, classical_diff_eq, 1e-4, t)
# a_p = np.real(PCsolver([1e-4, 1e-4], 1.001, fractional_diff_eq, a0, a1, num_points=len(t)))

fig, axs = plt.subplots(1, 1)

alphas = np.array([1.05, 1.5, 1.98])

cmap = plt.get_cmap("plasma_r", len(alphas))
norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(sm, ax=axs, label=r"$\zeta$")

i = 0
for alpha in alphas:
    if np.isclose(alpha, 1, 1e-2):
        a = caputoEuler(0.9, fractional_diff_eq, 1e-4, t)
    elif alpha < 1:
        a = caputoEuler(alpha, fractional_diff_eq, 1e-4, t)
    elif alpha > 1:
        a = np.real(
            PCsolver([1e-4, 1e-4], float(alpha), fractional_diff_eq, a0, a1, num_points=len(t)))
    axs.loglog(t, a, color=cmap(i))
    i += 1

a = caputoEuler(0.98, classical_diff_eq, 1e-4, t)
axs.loglog(t, a, linestyle="--")

axs.set_xlabel(r"$t$")
axs.set_ylabel(r"$a$")
plt.show()
