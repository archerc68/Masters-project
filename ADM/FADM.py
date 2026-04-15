import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Symbols
t, τ, alpha = sp.symbols('t τ alpha', positive=True)
λ = sp.symbols('λ')

# Fractional integral operator J^alpha
def J_alpha(g):
    kernel = (t - τ)**(alpha - 1) / sp.gamma(alpha)
    return sp.simplify(sp.integrate(kernel * g.subs(t, τ), (τ, 0, t)))

# Adomian polynomial generator
def adomian_polynomials(f, y_terms, N):
    yλ = sum(y_terms[k] * λ**k for k in range(N))
    fλ = f(yλ)
    A = []
    for n in range(N):
        An = sp.diff(fλ, λ, n).subs(λ, 0) / sp.factorial(n)
        A.append(sp.simplify(An))
    return A

# ---- FRACTIONAL ADM IMPLEMENTATION ----

# Number of terms to compute
N = 8

# Placeholder y_k symbols
y_syms = sp.symbols(f"y0:{N}")

# Initial y0(t)
y_exprs = [1]

# Define nonlinearity f(y)
def f(u):
    return u * sp.sqrt(0.2 * u ** (-3) + 1e-4 * u ** (-4))

# Build Adomian polynomials in terms of y0, y1, ..., y_{N-1}
A_syms = adomian_polynomials(f, y_syms, N)

# Now substitute progressively:
for n in range(N-1):

    # Build a substitution map {y0: y_exprs[0], y1: y_exprs[1], ...}
    subs_map = {y_syms[k]: y_exprs[k] for k in range(n+1)}
    
    # Substitute real expressions into A_n
    A_n_expr = sp.simplify(A_syms[n].subs(subs_map))
    
    # Compute y_{n+1} = J^alpha[A_n]
    y_next = sp.simplify(J_alpha(A_n_expr))
    
    # Append to list of computed y's
    y_exprs.append(y_next)

    # y(t)
    y_sum = sp.simplify(sum(y_exprs))

# Print results
for i, yi in enumerate(y_exprs):
    print(f"y{i}(t) =", yi)

y = sp.lambdify(t, y_sum.subs(alpha, 0.5))

plt.figure()
x = np.linspace(1e-6, 2, 100)
plt.plot(x, np.real(y(x)))
plt.show()