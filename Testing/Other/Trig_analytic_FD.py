import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymittagleffler import mittag_leffler


def D_alpha_sin(t, omega, alpha):
    f1 = mittag_leffler(1j * omega * t, 1, 1 - alpha)
    f2 = mittag_leffler(-1j * omega * t, 1, 1 - alpha)

    return np.real((f1 - f2) / (2j * t**alpha))


def D_alpha_cos(t, omega, alpha):
    f1 = mittag_leffler(1j * omega * t, 1, 1 - alpha)
    f2 = mittag_leffler(-1j * omega * t, 1, 1 - alpha)

    return np.real((f1 + f2) / (2 * t**alpha))


discontinuity = 1
variation = 0


if discontinuity:
    tvals = np.linspace(0.5, 2*np.pi, 250)

    plt.plot(tvals, D_alpha_cos(tvals, 1, 0.5) - np.cos(tvals + np.pi/4))
    plt.show()


if variation:
    tvals = np.linspace(0.125, 3 * np.pi, 250)

    plots = int(500)
    alphas = np.linspace(0, 2, plots)

    _, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    cmap = plt.get_cmap("plasma_r", plots)

    for i in range(plots):
        axs[0].plot(tvals, D_alpha_sin(tvals, 1, alphas[i]), color=cmap(i))

    for i in range(plots):
        axs[1].plot(tvals, D_alpha_sin(tvals, 2, alphas[i]), color=cmap(i))

    norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(
        sm,
        ticks=np.linspace(alphas[0], alphas[-1], 5),
        ax=axs,
        label=r"$\alpha$",
    )

    axs[0].set_title(r"Fractional derivatives of $\cos{(\omega x)}$")
    axs[0].set_ylabel(r"$D^{\alpha}\cos{(x)}$")

    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$D^{\alpha}\cos{(2x)}$")

    plt.savefig("Cosine_FD.png", dpi=1000)
    plt.show()
