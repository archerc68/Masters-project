import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from pymittagleffler import  mittag_leffler


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

    r = 4
    j = np.linspace(0, 1, N + 1)
    t = j**r

    # a and b coefficients
    def a(j, n):
        coeff = N ** (-r * alpha) / (alpha * (1 + alpha))

        if j == 0:
            term = (
                (n + 1) ** (r * alpha) * (alpha + 1)
                + ((n + 1) ** r - 1) ** (alpha + 1)
                - (n + 1) ** (r * (alpha + 1))
            )

        elif j == n + 1:
            term = ((n + 1) ** r - n**r) ** alpha

        else:
            term = (
                ((n + 1) ** r - (j - 1) ** r) ** (alpha + 1)
                - ((n + 1) ** r - j**r) ** (alpha + 1)
            ) / (j**r - (j - 1) ** r)
            +(
                ((n + 1) ** r - (j + 1) ** r) ** (alpha + 1)
                - ((n + 1) ** r - j**r) ** (alpha + 1)
            ) / ((j + 1) ** r - j**r)

        return coeff * term

    def b(j, n):
        coeff = N ** (-r * alpha) / alpha
        term = ((n + 1) ** r - j**r) ** alpha - ((n + 1) ** r - (j + 1) ** r) ** alpha
        return coeff * term

    y = np.empty((N + 1, d))
    y[0] = y_0

    for n in range(N):
        # Predictor
        b_sum = np.zeros_like(y_0, dtype=float)
        for j in range(0, n + 1):
            b_sum += b(j, n) * f(y[j])
        p = y_0 + 1 / gamma(alpha) * b_sum

        # Corrector
        a_sum = np.zeros_like(y_0, dtype=float)
        for j in range(0, n + 1):
            a_sum += a(j, n) * f(y[j])
        c = y_0 + 1 / gamma(alpha) * (a_sum + a(n + 1, n) * f(p))

        y[n + 1] = c

    if d == 1:
        return t, y
    else:
        return t, y[:, 0]


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

    t, y = PECE(q, y_0, d, g, T=T, N=N)

    return t, y


# Test case

# def f(x):
#     return -2*x

# y = PECE(0.5, 1, 1, f)
# t = np.linspace(0, 10, 501)

# plt.figure()
# plt.plot(t, y)
# plt.plot(t, mittag_leffler(-2*t**0.5, 0.5, 1))
# plt.show()

# Fractional Friedmann


def f(a, a_dot):
    return 1 * a_dot * np.sqrt(1e-4 * a ** (-4) + 0.2 * a ** (-3) + 0.8)


t, a = MultiPECE(1 / 2, 3, f, np.array([1e-6, 1e-6]), T=10, N=int(1e3))

fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")
axs.set_yscale("log")
# axs.set_ylim(1e-7, 10)

i = np.linspace(0, np.log1p(len(a) - 1), 250)
indices = np.unique(np.expm1(i).astype(int))

axs.plot(t, a)
plt.show()
