import numpy as np
import matplotlib.pyplot as plt

# Grünwald–Letnikov weights via recurrence
def gl_weights(alpha, N):
    k = np.arange(1, N+1)
    factors = -(alpha - (k-1)) / k
    w = np.empty(N+1)
    w[0] = 1.0
    w[1:] = np.cumprod(factors)
    return w

def fractional_solver(coeffs, powers, f, y0, T, h,
                      newton_maxit=20, newton_tol=1e-10):
    steps = int(T/h)
    t = h * np.arange(steps+1)
    y = np.empty(steps+1)
    y[0] = y0

    # Precompute weights
    W = [gl_weights(p, steps) for p in powers]
    scale = coeffs / (h**powers)
    C = np.sum(scale)  # since w0=1 for all

    for n in range(1, steps+1):
        # history convolution
        Hn = 0.0
        for s, w in zip(scale, W):
            Hn += s * np.dot(w[1:n+1], y[n-1::-1])

        # predictor
        y_guess = (f(y[n-1]) - Hn) / C

        # Newton iteration
        y_curr = y_guess
        for _ in range(newton_maxit):
            F = C*y_curr - f(y_curr) + Hn
            eps = 1e-8*(1+abs(y_curr))
            df = (f(y_curr+eps) - f(y_curr-eps))/(2*eps)
            dF = C - df
            step = -F/dF
            y_curr += step
            if abs(step) < newton_tol*(1+abs(y_curr)):
                break
        y[n] = y_curr
    return t, y

# ---- Example ----
coeff = np.array([1, 1, 1])   # coefficients
powers = np.array([2, 1, 0])  # orders
f = lambda y: 0 #-np.sin(y)
y0, T, h = 10.0, 10.0, 0.01

t, y = fractional_solver(coeff, powers, f, y0, T, h)

plt.figure(figsize=(9,4))
plt.plot(t, y, lw=1.5)
plt.title(f'Fractional ODE with coeff={coeff}, powers={powers}')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()