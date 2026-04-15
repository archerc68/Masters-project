import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


# -----------------------------
# f(y) and f'(y) for your case
# -----------------------------
def f(y):
    return 1e-4 * y ** (-4) + 0.2 * y ** (-3) + 0.8


def f_prime(y):
    return -4e-4 * y ** (-5) - 0.6 * y ** (-4)  # derivative of f(y)


# ------------------------------------------------------------
# Numerical solver for:   D^α y + y' f(y) = 0,   1 < α < 2
# ------------------------------------------------------------
def solve_fractional(alpha, y0, v0, T, N, tol=1e-12, max_iter=50):
    h = T / N
    x = np.linspace(0, T, N + 1)
    y = np.zeros(N + 1)

    # Initial conditions
    y[0] = y0
    y[1] = y0 + h * v0  # First-order Taylor

    # Precompute weights for L1 scheme
    a = np.array([(j + 1) ** (2 - alpha) - j ** (2 - alpha) for j in range(N)])
    C = 1.0 / (h**alpha * gamma(3 - alpha))

    # --------------------------
    # Main time-stepping loop
    # --------------------------
    for n in range(2, N + 1):
        # Build fractional convolution term
        frac = 0.0
        for j in range(n - 1):
            frac += a[j] * (y[n - j - 1] - 2 * y[n - j - 2] + y[n - j - 3])

        # Nonlinear solve using Newton iteration
        yn = y[n - 1]  # initial guess

        for k in range(max_iter):
            # Nonlinear equation F(yn) = 0
            F = C * frac + (yn - y[n - 1]) * f(yn) / h

            # Derivative dF/dy
            dF = (f(yn) + (yn - y[n - 1]) * f_prime(yn)) / h

            yn_new = yn - F / dF

            if abs(yn_new - yn) < tol:
                yn = yn_new
                break
            yn = yn_new

        y[n] = yn

    return x, y


x, y = solve_fractional(2, 0, 0.1, 10, 250, tol=1e-12, max_iter=50)

plt.figure()
plt.plot(x, y)
plt.show()
