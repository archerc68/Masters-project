from sympy import KroneckerDelta, ceiling, gamma, simplify, summation, symbols
from sympy.printing.latex import latex

i, j, k, N, n = symbols("i j k N n", positive=True, integer=True)
nu, L = symbols("nu L", positive=True)

num = (-1) ** (i - k) * i * gamma(i + k) * gamma(k - nu + 1 / 2)
den = (
    gamma(k + 1 / 2) * gamma(i - k + 1) * gamma(k - nu - j + 1) * gamma(k - nu + j + 1)
)
frac = num / den
sigma = 2 / ((1 + KroneckerDelta(j, 0)) * L**nu) * summation(frac, (k, ceiling(nu), i))

print(simplify(frac.subs(j, j + 1) / frac))
