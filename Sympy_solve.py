from sympy import KroneckerDelta, ceiling, gamma, simplify, summation, symbols, product
from sympy.printing.latex import latex


i, j, k, N, n = symbols("i j k N n", positive=True, integer=True)
nu, L = symbols("nu L", positive=True)

# n = ceiling(nu)

num = (-1) ** (i - k) * i * gamma(i + k) * gamma(k - nu + 1 / 2)
den = (
    gamma(k + 1 / 2) * gamma(i - k + 1) * gamma(k - nu - j + 1) * gamma(k - nu + j + 1)
)
frac = num / den
sigma = 2 / ((1 + KroneckerDelta(j, 0)) * L**nu) * summation(frac, (k, ceiling(nu), i))

# print(simplify(frac.subs(j, 0).subs(i, N)))

seed_num = (-1) ** (N - k) * N * gamma(N + k) * gamma(k - nu + 0.5)
seed_den = gamma(k + 0.5) * gamma(N - k + 1) * gamma(k - nu + 1) ** 2
seed = seed_num / seed_den


# print(simplify(simplify(frac.subs(i, n).subs(j, 0).subs(k, n))))
print(simplify(simplify(frac.subs(i, n).subs(j, 0).subs(k, n))))
