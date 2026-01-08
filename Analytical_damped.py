import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymittagleffler import mittag_leffler
from scipy.optimize import root_scalar

beta_185 = 1
var_plots = 0


def damped_driven(t, omega_0, y_0, y_dot_0, beta, g):
    omega_t_pow = -((omega_0 * t) ** beta)

    f1 = y_0 * mittag_leffler(omega_t_pow, beta, 1)
    f2 = y_dot_0 * t * mittag_leffler(omega_t_pow, beta, 2)
    f3 = t ** (beta - 1) * mittag_leffler(omega_t_pow, beta, beta)

    dt = t[1] - t[0]
    g1 = np.convolve(f3 * dt, g(t))[: len(f3)]

    return np.real(f1 + f2 + g1)


def damped(t, omega_0, y_0, y_dot_0, beta):
    omega_t_pow = -((omega_0 * t) ** beta)

    f1 = y_0 * mittag_leffler(omega_t_pow, beta, 1)
    f2 = y_dot_0 * t * mittag_leffler(omega_t_pow, beta, 2)

    return np.real(f1 + f2)


if beta_185:

    tvals = np.linspace(0, 50, int(1e4))

    dshm = damped(tvals, beta=1.85, omega_0=1, y_0=0.84, y_dot_0=0)

    plt.figure()
    plt.plot(tvals, dshm)
    plt.show()

    plt.figure()


if var_plots:
    plots = 1000
    tvals = np.linspace(0, 10, 250)
    betas = np.linspace(2, 1, plots)

    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    cmap = plt.get_cmap("plasma", plots)

    omega_0, y_0, y_dot_0 = 1, 1, 0

    for i in range(plots):
        dshm = damped(tvals, omega_0=omega_0, y_0=y_0, y_dot_0=y_dot_0, beta=betas[i])
        ax.plot(tvals, dshm, color=cmap(i))

    norm = mpl.colors.Normalize(vmin=betas[0], vmax=betas[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(
        sm,
        ticks=np.linspace(betas[0], betas[-1], 5),
        ax=ax,
        label=r"$^cD^{\beta} y(t) + \omega_0^{\beta}y(t)=0$",
    )

    plt.xlabel(r"t")
    plt.ylabel(r"$y(t)$")
    plt.title(
        r"$\omega_0 =$"
        + str(omega_0)
        + r"$, y(0) =$"
        + str(y_0)
        + r"$; \dot{y}(0)=$"
        + str(y_dot_0)
    )

    # plt.savefig("Betas_Analytic.png", dpi=1000)
    plt.show()
