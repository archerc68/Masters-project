import numpy as np
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler

y_0 = 0
y_dot_0 = 1
omega_0 = 1


def Fphase(t, alpha):
    y = y_dot_0 * t * mittag_leffler(-omega_0**2 * t ** alpha, 2 * alpha, 2)
    y_dot = y_dot_0* mittag_leffler(-omega_0**2 * t ** alpha, 2 * alpha, 1)
    return np.real(y), np.real(y_dot)

def Cphase(t, zeta):
    decay = np.exp(-zeta*omega_0*t)
    omega = omega_0**2 * np.sqrt(1 - zeta**2)

    c = np.cos(omega*t)
    s = np.sin(omega*t)

    const = (2*zeta*omega_0 + 1)*y_dot_0

    y = const * decay * s
    y_dot = const * decay * (c - zeta * omega_0 * s)

    return y, y_dot

ts = np.linspace(0, 4*np.pi, 500)

alphas = np.linspace(0.5, 1, 100)
FT, FA = np.meshgrid(ts, alphas)

zetas = np.linspace(0, 1, 100)
CT, CZ = np.meshgrid(ts, zetas)

Fphase_vec = np.vectorize(Fphase)
FY, FY_dot = Fphase_vec(FT, FA)

Cphase_vec = np.vectorize(Cphase)
CY, CY_dot = Cphase_vec(CT, CZ)

fig, (axs1, axs2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

axs1.plot_surface(CY, CY_dot, CZ, cmap="viridis")
axs2.plot_surface(FY, FY_dot, FA, cmap="inferno")

plt.show()
