import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymittagleffler import mittag_leffler

plt.rcParams["font.family"] = "Times New Roman"

spiral = 1
resonance = 0
var_plots = 0


def damped_driven(t, omega_0, y_0, y_dot_0, beta, g):
    omega_t_pow = -((omega_0 * t) ** beta)

    f1 = y_0 * mittag_leffler(omega_t_pow, beta, 1)
    f2 = y_dot_0 * t * mittag_leffler(omega_t_pow, beta, 2)
    f3 = t ** (beta - 1) * mittag_leffler(omega_t_pow, beta, beta)

    dt = t[1] - t[0]
    g1 = np.convolve(f3 * dt, g(t))[: len(f3)]

    return np.real(f1 + f2 + g1)


def damped(t, alpha, omega, q_0, p_0, m):
    omega2 = omega * omega
    omega_t_pow = -omega2 * t ** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, 1 + alpha)

    q = np.real(q_0 * f1 + p_0 / m * f2)
    p = np.real(p_0 * f1 - m * omega2 * q_0 * f2)

    return q, p


if spiral:
    tvals = np.linspace(0, 150, 2500)

    alpha1 = 1.85 / 2
    q1, p1 = damped(tvals, alpha1, 1, 1, 0, 1)

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].plot(q1, p1, color="blue")

    axs[0].set_xlim(-1.1, 1.1)
    axs[0].set_ylim(-1.1, 1.1)
    axs[0].set_aspect("equal")

    axs[0].set_title(r"$\alpha = $" + str(alpha1))
    axs[0].set_xlabel(r"$q_{\alpha}$")
    axs[0].set_ylabel(r"$p_{\alpha}$")

    alpha2 = 1.95 / 2
    q2, p2 = damped(tvals, alpha2, 1, 1, 0, 1)

    axs[1].plot(q2, p2, color="red")

    axs[1].set_xlim(-1.1, 1.1)
    axs[1].set_ylim(-1.1, 1.1)
    axs[1].set_aspect("equal")

    axs[1].set_title(r"$\alpha = $" + str(alpha2))
    axs[1].set_xlabel(r"$q_{\alpha}$")
    axs[1].set_ylabel(r"$p_{\alpha}$")

    plt.savefig("Spirals.png", dpi=1000)
    plt.show()

    Ham1 = 0.5 * np.hypot(p1, q1)
    Ham2 = 0.5 * np.hypot(p2, q2)

    plt.figure()

    plt.plot(tvals, Ham1, label=r"$\alpha = $" + str(alpha1), color="blue")
    plt.plot(tvals, Ham2, label=r"$\alpha = $" + str(alpha2), color="red")

    plt.xlabel(r"$t$")
    plt.ylabel(r"$\mathcal{H}_{\alpha}(t)$")
    plt.legend()

    plt.savefig("H_alpha.png", dpi = 1000)
    plt.show()

    def smooth(ang):    # Removes jumps in theta
        return np.radians(np.unwrap(ang))

    theta1 = smooth(np.arctan2(p1, q1))
    theta2 = smooth(np.arctan2(p2, q2))

    omega1 = np.abs(theta1/(tvals + 1e-8))
    omega2 = np.abs(theta2/(tvals + 1e-8))

    plt.figure()
    a = 50
    plt.plot(tvals[a:], omega1[a:], color="blue", label=r"$\alpha = $" + str(alpha1))
    plt.plot(tvals[a:], omega2[a:], color="red", label=r"$\alpha = $" + str(alpha2))

    plt.xlabel(r"$t$")
    plt.ylabel(r"$|\arccos{(p_{\alpha}/q_{\alpha})}/t|$")
    plt.legend()

    plt.savefig("Angle.png", dpi=1000)
    plt.show()

if resonance:
    tvals = np.linspace(0, 100, int(1e4))

    def g(t):
        return np.cos(2 * np.pi * 3e3 * t) + np.cos(2 * np.pi * 1e3 * t)

    dshm = damped_driven(tvals, beta=1.85, omega_0=1, y_0=0, y_dot_0=0, g=g)

    plt.figure()
    plt.plot(tvals, dshm)
    plt.show()


if var_plots:
    plots = 2000
    tvals = np.linspace(0, 10, 250)
    alphas = np.linspace(1, 0.5, plots)

    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    cmap = plt.get_cmap("plasma_r", plots)

    omega_0, y_0, y_dot_0 = 1, 1, 0

    for i in range(plots):
        dshm, _ = damped(
            tvals, omega=omega_0, q_0=y_0, p_0=y_dot_0, alpha=alphas[i], m=1
        )
        ax.plot(tvals, dshm, color=cmap(plots - i))

    norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(
        sm,
        ticks=np.linspace(1, 0.5, 5),
        ax=ax,
        label=r"$\alpha$",
    )

    plt.xlabel(r"t")
    plt.ylabel(r"$q_{\alpha}(t)$")
    plt.title(
        r"$\omega =$"
        + str(omega_0)
        + r"$, q_{\alpha}(0) =$"
        + str(y_0)
        + r"$; p_{\alpha}(0)/m=$"
        + str(y_dot_0)
    )

    plt.savefig("Alphas_Analytic.png", dpi=1000)
    plt.show()
