import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gamma


def GLkernal(n_max, h, alpha):
    k = np.arange(n_max + 1)
    rec = np.empty(n_max + 1)
    rec[0] = 1
    rec[1:] = 1 - (alpha + 1) / k[1:]
    kernal = np.cumprod(rec)

    return k, kernal*gamma(alpha + 1) / (h**alpha)


def GL(f, alpha, x):
    n = 250
    h = x[1] - x[0]
    F_j = f(x)
    _, G_k = GLkernal(n, h, alpha)
    return x[n:], fftconvolve(F_j, G_k, "valid")


def RLkernal(n, h, alpha):
    k = np.arange(0, n*h + h, h)
    return k, (h/gamma(alpha))*k**(alpha - 1)


def RLI(f, alpha, x):
    h = x[1] - x[0]
    F_k = f(x)
    _, R_k = RLkernal(len(x), h, alpha)
    conv = fftconvolve(F_k, R_k)[:len(x)]

    j = np.arange(0, len(x))
    conv -= 0.5*F_k[0]*(h**alpha)*(j**(alpha - 1))
    return x, h*conv/gamma(alpha)


alphas = np.linspace(1, 2, 5)

def f(x):
    return np.exp(2*x)

plt.figure()
x = np.linspace(0, 1, 1000)
for i in range(len(alphas)):
    x, y = GL(f, float(alphas[i]), x)
    plt.plot(x, y, label=str(alphas[i]))
plt.legend()
plt.show()
