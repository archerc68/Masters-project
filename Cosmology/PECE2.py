import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from pymittagleffler import mittag_leffler

# mu = 2

# D^{1.5} a = f(t, a, \dot{a}) \impies q = 1/2 \implies d = 3 (min 1)
# D^{1.75} a = f(t, a, \dot{a}) \implies q = 1/4 \implies d = 7 (min 2)
# D^{1.85} a = f(t, a, \dot{a}) \implies q = 1/20 \implies d = 37 (min 10)
# D^{1.9} a = f(t, a, \dot{a}) \implies q = 1/20 \implies d = 37 (min 10)

# a(0) = a_0 (=0), \dot{a}(0) = a_dot_0

# Transformed into D^q Y(x) = g(x, Y(x))

# Y := (y_0, y_1, ..., y_{d-1})^T
# g(Y) := (y_1, y_2, ..., y_{d-1}, f(y_0, y_{\alpha1/q}), ..., y_{alpha})^T


# PECE
def PECE(alpha, y_0, d, f, T=1, N=int(500)):
    h = T / N
    k = np.arange(1, N + 1)

    # a and b coefficients
    b = k**alpha - (k - 1) ** alpha
    a = (k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)

    y = np.zeros((N + 1, d))
    y[0] = y_0

    for j in range(1, N + 1):
        # Predictor
        b_sum = np.zeros(d)
        for k in range(j):
            b_sum += b[j - k - 1] * f(y[k])
        p = y_0 + h**alpha * b_sum / gamma(alpha + 1)

        # Corrector
        a_sum = np.zeros(d)
        for k in range(1, j):
            a_sum += a[j - k - 1] * f(y[k])
        a_const = (j - 1) ** (alpha + 1) - (j - 1 - alpha) * j**alpha
        y[j] = y_0 + h**alpha / gamma(alpha + 2) * (f(p) + a_const * f(y_0) + a_sum)

    # Allowing for standard PECE (0<\alpha<1 only)
    if d == 1:
        return y
    else:
        return y[:, 0]


def MultiPECE(q, d, f, y_bc, T=1, N=500):

    # y_0
    y_0 = np.zeros(d)
    jq = np.arange(d) * q

    mask = np.equal(jq, np.floor(jq))  # indices of rational numbers of jq
    j_bc = np.arange(d)[mask]  # j values

    y_0[j_bc] = y_bc

    print(y_0)

    # g(Y)

    def g(Y):
        gvec = np.zeros_like(Y)
        gvec[:-1] = Y[1:]  # Shifting y_0 -> y_1 etc.
        gvec[-1] = f(Y[0], Y[int(1 / q)])  # Last value of g(Y)
        return gvec

    print(g(y_0))

    return PECE(q, y_0, d, g, T=T, N=N)


# Test case

# def f(x):
#     return -2*x

# y = PECE(0.5, 1, 1, f, 10)
# t = np.linspace(0, 10, 501)

# plt.figure()
# plt.plot(t, y)
# plt.plot(t, mittag_leffler(-2*t**0.5, 0.5, 1))
# plt.show()

# Fractional Friedmann


def f(a, a_dot):
    aSoft = a
    return 1 * a_dot * np.sqrt(1e-4 * aSoft ** (-4) + 0.2 * aSoft ** (-3) + 0.8)

a = MultiPECE(1 / 2, 3, f, np.array([1e-4, 1e-4]), T=10, N=int(1e3))

fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")
axs.set_yscale("log")
# axs.set_ylim(1e-7, 10)

i = np.linspace(0, np.log1p(len(a) - 1), 250)
indices = np.unique(np.expm1(i).astype(int))


axs.plot(np.linspace(1e-7, 1, len(indices)), a[indices])
plt.show()
