import numpy as np

def compute_GL_weights(alpha, N):
    """Compute GL binomial weights via recurrence."""
    factors = np.empty(N + 1)
    factors[0] = 1.0
    k = np.arange(1, N + 1)
    factors[1:] = (alpha - (k - 1)) / k
    w = np.cumprod(factors)
    return w

def solve_nonlinear_fixed_point(rhs_func, initial_guess, tol=1e-10, max_iter=100):
    """Simple fixed-point iteration to solve y = rhs_func(y)."""
    y = initial_guess
    for _ in range(max_iter):
        y_new = rhs_func(y)
        if abs(y_new - y) < tol:
            return y_new
        y = y_new
    raise RuntimeError("Fixed-point iteration did not converge")

def solve_caputo_fde(alpha, f, y0, T, N):
    """Solve D_t^alpha y(t) = f(t, y(t)) with Caputo derivative."""
    dt = T / N
    t = np.linspace(0, T, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    w = compute_GL_weights(alpha, N)

    for n in range(1, N + 1):
        # Compute Caputo derivative via convolution
        D_alpha_y = 0.0
        for k in range(n):
            D_alpha_y += w[k] * (y[n - k] - y[n - k - 1])
        D_alpha_y /= dt**alpha

        # Implicit solve: D_alpha_y = f(t[n], y[n])
        rhs = lambda yn: f(t[n], yn)
        y[n] = solve_nonlinear_fixed_point(lambda yn: D_alpha_y - rhs(yn) + yn, y[n - 1])

    return t, y

def f_linear(t, y):
    return -y + np.sin(t)

alpha = 0.5
y0 = 1.0
T = 10.0
N = 1000

t, y = solve_caputo_fde(alpha, f_linear, y0, T, N)

import matplotlib.pyplot as plt
plt.plot(t, y)
plt.title(f"Caputo FDE Solution (α = {alpha})")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.show()