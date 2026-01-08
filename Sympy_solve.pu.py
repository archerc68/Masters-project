import matplotlib.pyplot as plt
import numpy as np
from sympy import simplify, symbols, 
from sympy.printing.latex import latex
from sympy.utilities.lambdify import lambdify
from sympy.abc import s, t

beta = symbols("beta", positive=True)
omega_0, y_0, y_dot_0 = symbols("omega_0 y_0 y_dot_0", real=True)

sfn = (s ** (1 - beta) * y_0 + s ** (beta - 2) * y_dot_0) / (s**beta + omega_0**2)

A = 1 / s

big_exp = inverse_laplace_transform(A, s, t)


print("Big expression:")
print(latex(simplify(big_exp)))

# f = lambdify((t, beta, omega_0, y_0, y_dot_0), big_exp, "mpmath")


# def damped(x):
#     return f(x, beta=1.85, omega_0=1, y_0=1, y_dot_0=0)


# t = np.linspace(0, 25, 250)
# ys = np.empty_like(t)
# for i in range(len(t)):
#     ys[i] = np.float64(damped(t[i]))

# plt.figure()
# plt.plot(t, ys)
# plt.show()
