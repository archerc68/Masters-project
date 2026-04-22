import numpy as np
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler

tvals = np.linspace(0, 1000, 25000)

alpha = 0.98

omega = 20

def damped(t, omega, alpha, q_0, p_0, m):

    omega_t_pow = - omega ** 2 * t** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = t ** alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

    if alpha > 0.5:
        q = np.real(q_0 * f1 + p_0/m * f2)
        p = np.real(p_0 * f1 - m * omega **2 * q_0 * f2)
    else:
        q = np.real(q_0 * f1)
        p = np.real(- m * omega **2 * q_0 * f2)

    return q, p

q, p = damped(tvals, omega, alpha, q_0=1, p_0=0, m=1)

plt.figure()
plt.plot(p, q)
plt.show()