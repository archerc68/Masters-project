import matplotlib.pyplot as plt
import numpy as np
from pymittagleffler import mittag_leffler
from scipy.optimize import curve_fit


# Boundary conditions
t = np.linspace(0, 20, 20)
omega = 1
m = 1
y0 = 1
ydot0 = 0


# Classical pendulum
def Cpend(t, gamma):

    OmegaDamp = np.sqrt(omega**2 + gamma**2)

    decay = np.exp(-gamma * t)

    f1 = y0 * np.cos(OmegaDamp * t)
    f2 = (ydot0 / m + gamma * y0) / OmegaDamp * np.sin(OmegaDamp * t)

    y = decay * (f1 + f2)

    return y


# Classical plot
gammas = np.linspace(0, 1, 20)

plt.figure()
ax = plt.axes(projection="3d")
x, y = np.meshgrid(t, gammas)
ax.plot_wireframe(x, y, Cpend(x, y))

ax.set_xlabel("t")
ax.set_ylabel(r"$\gamma$")
ax.set_zlabel("y")

plt.show()


# Fractional pendulum
def Fpend(t, alpha):

    omega_t_pow = -(omega**2) * t ** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = np.zeros_like(t)
    if not (t == 0 and alpha == 1):
        f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

    y = np.real(y0 * f1 + ydot0 / m * f2)

    return y


vfpend = np.vectorize(Fpend)


# Fractional plot
alphas = np.linspace(0.5, 1, 20)

plt.figure()
ax = plt.axes(projection="3d")
x, y = np.meshgrid(t, alphas)
ax.plot_wireframe(x, y, vfpend(x, y))

ax.set_xlabel("t")
ax.set_ylabel(r"$\alpha$")
ax.set_zlabel("y")

plt.show()
