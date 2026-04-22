import numpy as np
import matplotlib.pyplot as plt
from fodeint import caputoEuler

t = np.linspace(1e-4, 100, 2500)


def fractional_diff_eq(a, t):
    return a * np.sqrt(0.2 * a ** (-3) + 1e-4 * a ** (-4) + 0.8)

def classical_diff_eq(a, t):
    return a * np.sqrt(0.2 * a ** (-3) + 1e-4 * a ** (-4) + 0.8)


a = caputoEuler(0.1, fractional_diff_eq, 1e-4, t)
a_0 = caputoEuler(0.999, classical_diff_eq, 1e-4, t)

plt.figure()
plt.loglog(t, a)
plt.loglog(t, a_0)
plt.show()
