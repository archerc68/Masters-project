from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np


# Nth order derivative (GL w/ integar alpha)
def differentiate(f, N, t):
    dt = 1e-2

    # k values
    k = np.arange(N + 1)

    # a_k coefficients ((-1)^k chose alpha k)
    a_k = np.empty(N + 1)
    a_k[0] = 1
    a_k[1:] = 1 - (N + 1) / k[1:]
    a_k = np.cumprod(a_k)

    # t values for f_k := f(t_k)
    t_k = t[:, None] + (N - k[None, :]) * dt

    # dot product between coefficent vector and f(t values)
    return f(t_k) @ a_k / dt**N



# # Grünwald–Letnikov FD
def GL(f, alpha, t):
    limit = int(1e3)
    dt = 1e-2

    # k values
    k = np.arange(limit + 1)

    # a_k coefficients ((-1)^k chose alpha k)
    a_k = np.empty(limit + 1)
    a_k[0] = 1
    a_k[1:] = 1 - (alpha + 1) / k[1:]
    a_k = np.cumprod(a_k) / dt**alpha

    # t values for f_k := f(t_k)
    t_k = t[:, None] - k[None, :] * dt

    f_k = f(t_k)

    # dot product between coefficent vector and f(t values)
    return f_k @ a_k

# Grünwald–Letnikov FD
# def GL(f, alpha, t):
#     n = len(t)
#     limit = min(n - 1, int(1e3))
#     dt = t[1] - t[0]

#     #k values
#     k = np.arange(limit + 1)

#     #a_k coefficients ((-1)^k chose alpha k)
#     a_k = np.empty(limit + 1)
#     a_k[0] = 1
#     a_k[1:] = 1 - (alpha + 1) / k[1:]
#     a_k = np.cumprod(a_k) 

#     f_k = f(t)

#     return convolve1d(f_k, a_k)[:n] / dt**alpha


def RLI(f, alpha, x, a=0):
    dx, n = x[1]-x[0], len(x)
    fvals = f(x)

    # Kernel sampled in grid units
    u = np.arange(n) * dx
    gvals = u**(alpha-1) / gamma(alpha)
    gvals[0] = 0  # define 0^(alpha-1) = 0 for alpha>0

    out = fftconvolve(fvals, gvals, mode="full")[:n] * dx

    return out

def RL(f, alpha, x):

    dx = x[1] - x[0]

    n = int(np.ceil(alpha))
    integrand = RLI(f, alpha - n + 1, x)
    n += 1
    
    k = np.arange(n + 1)

    kernal = np.empty(n + 1)
    kernal[0] = 1
    kernal[1:] = 1 - (n + 1) / k[1:]
    kernal = np.cumprod(kernal)/dx**n

    return x[n:len(x) - n], convolve1d(integrand, kernal)[n:len(integrand)-n]


# Generalised Leibenez rule
def Leibenez(f, g, alpha, D, t):
    limit = 100  # Resolution

    result = 0

    # k values
    k = np.arange(1, limit + 1)

    # a_k coefficients ((-1)^k chose alpha k)
    factors = np.empty(limit + 1)
    factors[0] = 1
    factors[1:] = (alpha - k) / k + 1
    chose = np.cumprod(factors)

    for i in range(limit):
        result += chose[i] * D(f, i, t) * D(g, alpha - i, t)

    return result


def main():
    # Inspecting functions
    def f(x):
        return np.cos(x)
    
    dx = 1e-2
    T = 2*np.pi
    xs = np.arange(int(T/dx))*dx

    plt.figure()
    num = 49
    for i in range (num):
        D = GL(f, 2*i/num, xs)
        plt.plot(xs, D)
    plt.show()


main()
