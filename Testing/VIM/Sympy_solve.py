import matplotlib.pyplot as plt
import numpy as np
from sympy import diff, integrate, lambdify, simplify, sqrt, symbols, solve

a, t, x, alpha, omega_m, omega_r, eps = symbols(
    "a, t, x alpha omega_m omega_r eps", positive=True
)

omega_m, omega_r = 0.27, 8.27e-5


Nu = diff(a, t) - a * (omega_m * a ** (-3) + omega_r * a ** (-4)) ** 0.5
Lambda = eps - t

print("n = 0:")
a = t
print(a)

for n in range(1, 10):
    print("n = " + str(n) + ":")
    a = a - integrate(Lambda * (Nu.subs(t, eps)), (eps, 0, t))
    a = simplify(a.subs(a, a))
    print("a = " + str(a))


print(solve(a, t))

a_func = lambdify(t, a_pred, "numpy")

print(a_func(1))
ts = np.linspace(0.1, 1)
plt.figure()
plt.plot(ts, a_func(ts))
plt.show()
