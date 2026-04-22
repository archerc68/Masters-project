import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler


def w2(alpha, k, n):
    p = 2 - alpha  # power
    if k == -1:
        return 1
    elif k == 0:
        return 2**p - 3
    elif 1 <= k <= n - 2:
        return (k + 2) ** p - 3 * (k + 1) ** p + 3 * k**p - (k - 1) ** p
    elif k == n - 1:
        return -2 * (n**p) + 3 * (n - 1) ** p - (n - 2) ** p
    elif k == n:
        return n**p - (n - 1) ** p


def L2(f, x):
    N = len(x)
    h = x[1] - x[0]
    a = x[0]
    alpha = 1.5

    f_c = np.empty_like(x)

    for n in range(N):
        f_cn = 0.0
        for k in range(-1, n):
            f_cn += w2(alpha, k, n) * f(a + h * (n - k))
        f_c[n] = f_cn

    return h ** (-alpha) / gamma(3 - alpha) * f_c


def w1(alpha, mu, n, h):
    p = 1 - alpha
    return h ** (-alpha) * ((n - mu) ** p - (n - mu - 1) ** p)


def L1_a2(func, x, f_dot_0):
    f = func(x)
    h = x[1] - x[0]
    alpha = 1.5

    # First derivative of f(x)
    f_dot = np.zeros_like(x)
    f_dot[0] = f_dot_0
    for i in range(1, len(x)):
        f_dot[i] = (f[i] - f[i - 1]) / h

    # L1 scheme for alpha-1
    df = np.zeros_like(x)
    for n in range(len(x)):
        df_sub = 0.0
        for j in range(n):
            df_sub += w1(alpha - 1, n - j - 1, n, h) * (f_dot[n - j] - f_dot[n - j - 1])
        df[n] = df_sub
    return df / gamma(3 - alpha), f_dot


def f(x):
    return np.sin(x)


def D_alpha_sin(t, alpha=1.5, omega=1):
    omega_t = omega * t * 1j
    den = 2j * t**alpha
    num = mittag_leffler(omega_t, 1, 1 - alpha) - mittag_leffler(-omega_t, 1, 1 - alpha)
    return np.real(num / den)


def b(alpha, j, h):
    sigma = 1 - alpha / 2
    p = 2 - alpha
    return (
        h ** (1 - alpha) / gamma(3 - alpha) * ((j + 1 - sigma) ** p - (j - sigma) ** p)
    )


def l2_1sigma(func, x, f_dot0):
    f = func(x)
    h = x[1] - x[0]
    alpha = 1.5

    # First derivative of f(x)
    f_dot = np.zeros_like(x)
    f_dot[0] = f_dot0
    for i in range(1, len(x)):
        f_dot[i] = (f[i] - f[i - 1]) / h

    df = np.zeros_like(x)
    for n in range(len(x)):
        df_sub = 0.0
        for k in range(1, n):
            df_sub += b(alpha, n - k, h) * (f_dot[k] - f_dot0)
        df[n] = df_sub
    return df


# Plotting
fig, ax = plt.subplots(1, 1)

x = np.linspace(0, 50, 250)
y_2 = L2(f, x)
y_1, ydot = L1_a2(f, x, 1)
y = D_alpha_sin(x)
y_21sigma = l2_1sigma(f, x, 1)

ax.plot(x, y, color="black", linestyle="--", label="True")
ax.plot(x, y_2, label="L2")
ax.plot(x, y_1, label="L1")
ax.plot(x, y_21sigma, label=r"$L2-1_{\sigma}$")
ax.legend()
ax.set_ybound(-2, 2)
plt.show()
