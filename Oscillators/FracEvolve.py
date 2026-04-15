import numpy as np
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler


# Pendulum params
omega_0 = 1
tvals = np.linspace(0, 10, 250)
q_0 = 1
p_0 = 0
m = 1


def Fdamp(t, alpha):

    omega_t_pow = -(omega_0**2) * t ** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

    if alpha > 0.5:
        q = np.real(q_0 * f1 + p_0 / m * f2)
    else:
        q = np.real(q_0 * f1)
    return q


def Cdamp(x, gamma):
    decay = np.exp(-gamma * x)

    if np.isclose(omega_0, gamma):
        oscillation = q_0 * np.cos(omega_0 * x)
    else:
        omega = np.sqrt(np.absolute(omega_0**2 - gamma**2))
        sfactor = (p_0 + gamma * q_0) / omega
        oscillation = np.cos(omega * x) + sfactor * np.sin(omega * x)
    return decay * oscillation


fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

alphas = np.array([1, 0.85, 0.5])
gammas = np.array([0, 0.5, 1.1])

i = 0
for gamma in gammas:
    axs[0, i].plot(tvals, Cdamp(tvals, gamma))
    axs[0, i].annotate(r"$\gamma =$ " + str(gamma), (5.5, 0.9))
    i += 1

i = 0
for alpha in alphas:
    axs[1, i].plot(tvals, Fdamp(tvals, alpha), color="orange")
    axs[1, i].annotate(r"$\alpha =$ " + str(alpha), (5.5, 0.9))
    i += 1

axs[1, 1].set_xlabel("t")
axs[0, 0].set_ylabel("y(t)")
axs[1, 0].set_ylabel("y(t)")

plt.savefig("Figures/FacClasBox.png", dpi=100)
plt.show()
