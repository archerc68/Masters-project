import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit()
def gl_weights(alpha, N):
    k = np.arange(1, N+1, dtype=np.float64)
    factors = -(alpha - (k-1)) / k
    w = np.empty(N+1, dtype=np.float64)
    w[0] = 1.0
    w[1:] = np.cumprod(factors)
    return w

@njit()
def fractional_solver_numba(scale, W, f, y0, steps, h,
                            newton_maxit=20, newton_tol=1e-10):
    m = scale.shape[0]
    t = h * np.arange(steps+1, dtype=np.float64)
    y = np.empty(steps+1, dtype=np.float64)
    y[0] = y0
    conv = np.zeros(m, dtype=np.float64)
    C = np.sum(scale)

    for n in range(1, steps+1):
        Hn = 0.0
        for i in range(m):
            conv[i] += W[i, n] * y[n-1]
            Hn += scale[i] * conv[i]

        # predictor
        y_guess = (f(y[n-1]) - Hn) / C

        # Newton iteration
        y_curr = y_guess
        for _ in range(newton_maxit):
            F = C*y_curr - f(y_curr) + Hn
            eps = 1e-8*(1+abs(y_curr))
            df = (f(y_curr+eps) - f(y_curr-eps)) / (2*eps)
            dF = C - df
            step = -F/dF
            y_curr += step
            if abs(step) < newton_tol*(1+abs(y_curr)):
                break
        y[n] = y_curr
    return t, y

# ---- Example usage ----
coeff = np.array([1, 1, 1])   # coefficients
powers = np.array([2, 1, 0])  # orders
y0, T, h = 10.0, 10.0, 0.01
steps = int(T/h)

# Precompute weights and scales
W = np.empty((len(powers), steps+1), dtype=np.float64)
for i, alpha in enumerate(powers):
    W[i] = gl_weights(alpha, steps)
scale = coeff / (h**powers)

# Define RHS as a numba-compatible function
@njit
def f(y):
    return 0 #np.sin(y)

t, y = fractional_solver_numba(scale, W, f, y0, steps, h)

plt.plot(t, y)
plt.xlabel("t"); plt.ylabel("y(t)")
plt.title("Numba-friendly fractional ODE solver")
plt.grid(True); plt.show()