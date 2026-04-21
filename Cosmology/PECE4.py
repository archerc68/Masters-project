import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from pymittagleffler import mittag_leffler
from numba import njit
from time import time


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

    r = 2
    j = np.linspace(0, 1, N + 1)
    t = T * (j**r)

    print(t[1])
    print(t[-1])

    # a and b coefficients
    @njit()
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

    @njit()
    def b(j, n):
        coeff = N ** (-r * alpha) / alpha
        term = ((n + 1) ** r - j**r) ** alpha - ((n + 1) ** r - (j + 1) ** r) ** alpha
        return coeff * term

    y = np.empty((N + 1, d))
    y[0] = y_0

    ga = gamma(alpha)
    @njit()
    def evolve(y):
        for n in range(N):
            # Predictor
            b_sum = np.zeros(d)
            for j in range(0, n + 1):
                b_sum += b(j, n) * f(y[j])
            p = y_0 + 1 / ga * b_sum

            # Corrector
            a_sum = np.zeros(d)
            for j in range(0, n + 1):
                a_sum += a(j, n) * f(y[j])
            c = y_0 + 1 / ga * (a_sum + a(n + 1, n) * f(p))

            y[n + 1] = c
        return y

    y = evolve(y)

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

    index = int(1 / q)
    @njit()
    def g(Y):
        gvec = np.zeros_like(Y)
        gvec[:-1] = Y[1:]  # Shifting y_0 -> y_1 etc.
        gvec[-1] = f(Y[0], Y[index])  # Last value of g(Y)
        return gvec

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

# Parameters

a_0 = 1e-6
a_dot_0 = 1e-6

alpha = 1
H_tilde0 = 70
Omega_r0 = 1e-4
Omega_m0 = 0.2
Omega_lambda = 0.8

a_dot_0 = 1e-4

a_eq = Omega_r0 / Omega_m0

gam = 1.5

@njit()
def f(a, a_dot):
    aSoft = a + 1e-6
    return (
        alpha ** (1 / 2)
        * H_tilde0 ** (gam - 1)
        * a_dot
        * np.sqrt(Omega_r0 * aSoft ** (-4) + Omega_m0 * aSoft ** (-3) + Omega_lambda)
    )


start = time()
t, a = MultiPECE(1 / 2, 3, f, np.array([1e-4, 1e-9]), T=1e-1, N=int(1e4))
end=time()

print(end-start)


fig, axs = plt.subplots(1, 1)
axs.set_xscale("log")
axs.set_yscale("log")
# axs.set_ylim(1e-7, 10)

i = np.linspace(0, np.log1p(len(a) - 1), 250)
indices = np.unique(np.expm1(i).astype(int))

lb = np.where(t > 1e-6)

axs.plot(t, a)

# Hor
a_RM = Omega_r0 / Omega_m0
a_ML = (Omega_m0 / Omega_lambda) ** (1 / 3)

axs.axhline(a_RM, t[1], t[-1], linewidth=1, color="k")
axs.axhline(a_ML, t[1], t[-1], linewidth=1, color="k")

plt.show()


# History

# # qth derivative of radiation
# def rad(t, q, d):
#     coeff = (
#         (alpha * Omega_r0) ** (1 / 4)
#         * np.sqrt(-gamma(1 / 2 - gam / 2) / gamma(gam / 2 - 1 / 2))
#         * H_tilde0 ** ((gam - 1)/ 2)
#     )

#     rad_array = np.empty((len(t), d))

#     t_soft = t + 0   # Softened t parameter
#     for i in range(d):
#         fd_coeff = gamma((gam + 1) / 2) / gamma((gam + 1) / 2 - i*q)
#         tdep = t_soft ** ((gam - 1) / 2 - i*q)
#         rad_array[:, i] = fd_coeff * tdep

#     a_rad = coeff * rad_array
#     rad_dom = np.where(a_rad[:, 0] < Omega_r0 / Omega_m0)

#     print(Omega_r0 / Omega_m0)

#     return a_rad

# t = np.logspace(-10, -6, 3)
# a = rad(t, 0, 3)
# print(a)


