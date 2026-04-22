import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_func

alpha, Omega_R0, H_tilde0 = 1, 1e-4, 70
Omega_m0 = 0.2

t = np.logspace(-18, -8, 100)


def rad(gamma):
    B = (gamma - 1) / 2

    t_pow = (H_tilde0 * t) ** B

    B_coeff = np.sqrt(np.pi/(B*np.sin(np.pi*B)))/gamma_func(B)

    return (
        (alpha * H_tilde0) ** (1 / 4) * B_coeff * t_pow
    )

gammas = np.linspace(1.1, 2, 500)

cmap = plt.get_cmap("jet", len(gammas))

fig, axs = plt.subplots(1, 1)

axs.set_xscale("log")
axs.set_yscale("log")

for i in range(len(gammas)):
    axs.plot(t, rad(gammas[i]), color=cmap(i))

a_RM = Omega_R0/Omega_m0
axs.axhline(a_RM, 0, 1, linewidth=1, color="k", linestyle="--")

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(gammas[0], gammas[-1]))
fig.colorbar(sm, ax=axs, ticks=np.linspace(gammas[0], gammas[-1], 5), label=r"$\gamma$", orientation="vertical")

axs.text(1.1*t[0], 1.1*a_RM, r"$a_{RM}^{eq}$",
         horizontalalignment="left",
         verticalalignment="bottom")

axs.set_xlabel("t")
axs.set_ylabel("a(t)")

# plt.savefig("Figures/Radiation.svg")

print(1/H_tilde0 * (alpha*Omega_R0)**(-1/2) * (Omega_R0/Omega_m0)**2/2)

plt.show()