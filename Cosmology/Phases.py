import numpy as np
import matplotlib.pyplot as plt

Omega_R0, Omega_M0, Omega_L = 1e-4, 0.3, 0.7

H_tilde0 = 70
alpha=1

def f(a, gamma):
    return alpha*H_tilde0**(2*(gamma-1))*(Omega_R0*a**(-4) + Omega_M0*a**(-3) + Omega_L)

a_RM = Omega_R0/Omega_M0
a_ML = (Omega_L/Omega_M0)**(1/3)

a =np.logspace(-5, 2, 250)

fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")
axs.set_yscale("log")

axs.plot(a, f(a, 1.5))

axs.axvline(a_RM, 0, 1e100, linestyle="--", color="k")
axs.axvline(a_ML, 0, 1e100, linestyle="--", color="k")

plt.show()