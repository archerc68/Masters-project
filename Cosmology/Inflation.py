from pymittagleffler import mittag_leffler
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Constants

gamma = np.linspace(1, 2, 500)
B = (gamma - 1) / 2

alpha = 1
H_tilde0 = 70
Omega_phi = 0.5
kappa = np.sqrt(alpha * Omega_phi) * H_tilde0 ** (gamma - 1)

a_0 = 0
a_dot_0 = 1

# a(t)


def a(gamma, t):
    kappa = np.sqrt(alpha * H_tilde0 ** (2 * (gamma - 1)))

    tE = t * mittag_leffler(kappa * t ** ((gamma - 1)/2), (gamma - 1)/2, 2)

    return np.real(a_0 + a_dot_0 * tE)


def a_dot(gamma, t):
    kappa = np.sqrt(alpha * H_tilde0 ** (2 * (gamma - 1)))

    tE = mittag_leffler(kappa * t ** ((gamma - 1)/2), (gamma - 1)/2, 1)

    return np.real(a_dot_0 * tE)


fig, axs = plt.subplots(2, 2)

# Axis scaling

axs[0, 0].set_yscale("log")
axs[0, 1].set_yscale("log")
axs[1, 0].set_yscale("log")
axs[1, 1].set_yscale("log")

axs[0, 1].set_xscale("log")
axs[1, 1].set_xscale("log")

# Sharing axis

axs[1, 0].sharex(axs[0, 0])
axs[1, 1].sharex(axs[0, 1])

axs[0, 1].sharey(axs[0, 0])
axs[1, 1].sharey(axs[1, 0])

# Hiding ticks

axs[0, 0].tick_params(labelbottom=False)
axs[0, 1].tick_params(labelbottom=False, labelleft=False)
axs[1, 1].tick_params(labelleft=False)

# Formatting ticks

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))

axs[1, 0].xaxis.set_major_formatter(formatter)

# Colour bar

cmap = plt.get_cmap("jet", len(gamma))

# Plotting

tlin = np.linspace(0, 1e-3, 250)
tlog = np.logspace(-9, -3, 250)

for i in range(len(kappa)):
    # a(t)
    axs[0, 0].plot(tlin, a(gamma[i], tlin), color=cmap(i))
    axs[0, 1].plot(tlog, a(gamma[i], tlog), color=cmap(i))

    # a_dot(t)
    axs[1, 0].plot(tlin, a_dot(gamma[i], tlin), color=cmap(i))
    axs[1, 1].plot(tlog, a_dot(gamma[i], tlog), color=cmap(i))

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(gamma[0], gamma[-1]))
fig.colorbar(sm, ax=axs, ticks=np.linspace(gamma[0], gamma[-1], 5), label=r"$\gamma$", orientation="horizontal")

# axs[:, :].set_xlabel("t")

axs[0, 0].set_ylabel(r"$a_{\phi}(t)/\dot{a}(0)$")
axs[1, 0].set_ylabel(r"$\dot{a}_{\phi}(t)/\dot{a}(0)$")


fig.savefig("Figures/Inflation.svg")
plt.show()
