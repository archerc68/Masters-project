import numpy as np
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler
import matplotlib as mpl
import scipy

alpha=1
H_tilde0 = 2.27e-18

def H(gamma, t):
    kappa = np.sqrt(alpha * H_tilde0 ** (2 * (gamma - 1)))

    num = mittag_leffler(kappa * t ** ((gamma - 1)/2), (gamma - 1)/2, 1)

    den = t * mittag_leffler(kappa * t ** ((gamma - 1)/2), (gamma - 1)/2, 2)

    return np.real(num/den)


t = np.logspace(-4, 4, 100)
gamma = np.linspace(1, 2, 500)

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
fig.subplots_adjust(hspace=0)

cmap = plt.get_cmap("jet", len(gamma))

ax1.set_xscale("log")
ax1.set_yscale("log")

H_15 = H(1.5, t)

Hs = np.empty((len(gamma), len(t)))
for i in range(len(gamma)):
    Hs[i] = H(gamma[i], t)
H_mean = scipy.stats.pmean(Hs, axis=0)

for i in range(len(gamma)):
    H_i = Hs[i]
    ax1.plot(t, H_i, color=cmap(i))
    ax2.plot(t, H_i / H_15, color=cmap(i))

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(gamma[0], gamma[-1]))
fig.colorbar(sm, ax=(ax1, ax2), ticks=np.linspace(gamma[0], gamma[-1], 5), label=r"$\gamma$", orientation="vertical")

ax1.set_ylabel(r"$H_{\phi}(t, \gamma)$")
ax2.set_ylabel(r"$H_{\phi}(t, \gamma) / H_{\phi}(t, 1.5)$")
ax2.set_xlabel("t")

plt.show()