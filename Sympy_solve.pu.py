import matplotlib.pyplot as plt
import numpy as np
from sympy import cos, gamma, integrate, simplify, sin, symbols
from sympy.printing.latex import latex
from sympy.utilities.lambdify import lambdify

t, beta, omega_0, x, y_0, y_dot_0 = symbols("t beta omega_0 x y_0 y_dot_0")

cs = x ** (beta - 1) * sin(omega_0 * x)
ss = x ** (beta - 1) * cos(omega_0 * x)

cs_int = simplify(integrate(cs, (x, 0, t)))
ss_int = simplify(integrate(ss, (x, 0, t)))

A = (y_0 * cos(omega_0 * t) + (y_dot_0 / omega_0) * sin(omega_0 * t)) / gamma(2 - beta)
B = (y_0 * sin(omega_0 * t) - (y_dot_0 / omega_0) * cos(omega_0 * t)) / gamma(2 - beta)

big_exp = simplify(A * cs_int + B * ss_int)


print("Big expression:")
print(latex(big_exp))

f = lambdify((t, beta, omega_0, y_0, y_dot_0), big_exp, "mpmath")


def damped(x):
    return f(x, beta=1.85, omega_0=1, y_0=1, y_dot_0=0)


t = np.linspace(0, 10, 250)
ys = np.empty_like(t)
for i in range(len(t)):
    ys[i] = np.float64(damped(t[i]))

plt.figure()
plt.plot(t, ys)
plt.show()
