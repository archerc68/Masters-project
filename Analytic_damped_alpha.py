import numpy as np
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler

tvals = np.linspace(0, 20, 250)

alpha = 1.85/2

omega = 1

def damped(t, A, B):

    omega_t_pow = - omega ** 2 * t** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = t * mittag_leffler(omega_t_pow, 2 * alpha, 2)

    return np.real(A * f1 + B * f2)

plt.figure()
plt.plot(tvals, damped(tvals, 1, 0))
plt.show()