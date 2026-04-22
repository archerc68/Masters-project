import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from sympy import gamma

# Symbols
k_R, k_M, k_Lambda, a_dot_0, t_0, zeta, t, a = sp.symbols(
    "k_R k_M k_Lambda a_dot_0 t_0 zeta t a", positive=True
)

A = 6 * (gamma(zeta) - k_Lambda)
c = 6*(gamma(zeta)-k_Lambda) + 3 * k_M + 2*k_R - 6*gamma(zeta)*a_dot_0*t_0
B = 6 * gamma(zeta)*a_dot_0 * t + c
D = 3*k_M
E = 2*k_R

res = A * a**4 - B * a**3 + D * a + E

sol = sp.solve(res, a)

print("\n0:\n")
expr0, branch_cond0 = sol[0].args[0]
print(branch_cond0)
print(sp.simplify(expr0))

print("\n1:\n")
expr1, branch_cond1 = sol[1].args[0]
print(branch_cond1)
print(sp.simplify(expr1))

print("\n2:\n")
expr2, branch_cond2 = sol[2].args[0]
print(branch_cond2)
print(sp.simplify(expr2))

print("\n3:\n")
expr3, branch_cond3 = sol[3].args[0]
print(branch_cond3)
print(sp.simplify(expr3))
