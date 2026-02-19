import numpy as np
from scipy.integrate import cumulative_simpson
import matplotlib.pyplot as plt
from scipy.special import gamma

plot_basis = 1
plot_AB = 1
analytic = 1

beta = 1.95

# Special sine and cosine integrands


def ss(x):
    return x ** (beta - 1) * np.sin(x)


def cs(x):
    return x ** (beta - 1) * np.cos(x)


xvals = np.linspace(0, 20, 5000)

ss_int = cumulative_simpson(ss(xvals), x=xvals, initial=0)
cs_int = cumulative_simpson(cs(xvals), x=xvals, initial=0)

if plot_basis:
    plt.figure()
    plt.plot(xvals, ss_int, label=r"$\int_0^t t'^{\beta-1}\cos{(\omega_0 t')}dt'$")
    plt.plot(xvals, cs_int, label=r"$\int_0^t t'^{\beta-1}\sin{(\omega_0 t')}dt'$")

    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel("Integrand")
    plt.title(r"Damped basis fns. for $\beta=$" + str(beta))

    plt.show()

# A(t) and B(t)

y_0 = 1
y_dot_0 = 0
omega = 1


def AB(t):
    c, s = np.cos(omega * t), np.sin(omega * t)
    A = (y_0 * c + y_dot_0 / omega * s) / gamma(2 - beta)
    B = (y_0 * s - y_dot_0 / omega * c) / gamma(2 - beta)
    return A, B


if plot_AB:
    tvals = np.linspace(0, 10, 250)
    A, B = AB(tvals)
    plt.figure()
    plt.plot(tvals, A)
    plt.plot(tvals, B)
    plt.show()


if analytic:
    A, B = AB(xvals)
    y = A * cs_int + B * ss_int
    plt.figure()
    plt.plot(xvals, y)
    plt.show()
