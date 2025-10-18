import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve
from scipy.special import gamma


### Kernals ##
def GLkernal(n_max, h, alpha):
    k = np.arange(n_max)
    g_k = np.empty(n_max)
    g_k[0] = 1 / (h**alpha)
    g_k[1:] = 1 - (alpha + 1) / k[1:]

    return np.cumprod(g_k)


def RLkernal(n, h, alpha):
    k = np.arange(0, n)
    return ((h**alpha) / gamma(alpha)) * k ** (alpha - 1)


### GL ###
def GL(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    n = 1000
    G_k = GLkernal(n, h, alpha)
    F_j = f(x)
    return x[b:], fftconvolve(F_j, G_k)[b : len(x)]


### Riemann-Liouville ###


# RLI
def RLI(f, alpha, x):
    h = x[1] - x[0]
    F_k = f(x)
    R_k = RLkernal(len(x), h, alpha)
    N = len(x) - 1
    F_pad = np.pad(F_k, (0, N))
    R_pad = np.pad(R_k, (0, N))
    conv = np.real(ifft(fft(F_pad) * fft(R_pad)))[: len(x)]

    conv -= 0.5 * F_k[0] * R_k[0]
    return x, conv


# RL
def RL(f, alpha, x):
    def g(x):
        x, y = RLI(f, np.ceil(alpha) - alpha + 1, x)
        return y

    x, deriv = GL(g, np.ceil(alpha) + 1, x)
    return x, deriv


# RLI using ffts
def RLI_fft(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 1000  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    R_k = RLkernal(len(x), h, alpha)
    F_j = f(x)
    conv = fftconvolve(F_j, R_k)[: len(x)] - 0.5 * F_j[0] * R_k
    return x[b:], conv[b:]


# RL using ffts -- currently broken
def RL_fft(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    n = 250

    # Kernals
    R_k = RLkernal(len(x), h, np.ceil(alpha) - alpha + 1)
    G_k = GLkernal(n, h, np.ceil(alpha) + 1)
    F_j = f(x)

    N = 2 * len(x) + n - 1
    Rpad = np.pad(R_k, (0, N - len(x)))
    Gpad = np.pad(G_k, (0, N - n))
    Fpad = np.pad(F_j, (0, N - len(x)))

    temp1 = fft(Rpad) * fft(Gpad) * fft(Fpad)
    temp2 = 0.5 * F_j[0] * fft(Rpad) * fft(Gpad)

    conv = np.real(ifft(temp1 - temp2))

    return x[b:], conv[b : len(x)]


### Plotting ###


def main(FD):
    # Alpha values
    alphas = np.linspace(1, 3, 50)

    # Function
    def f(x):
        return np.cos(x)

    # Displaying plot
    plt.figure()
    for i in range(len(alphas)):
        x = np.arange(
            -2 * np.pi, 2 * np.pi, 1e-2
        )  # Increasing resolution may warrant an increase in GL buffer size
        x, y = FD(f, float(alphas[i]), x)

        plt.plot(x, y, label=str(alphas[i]))
    # plt.legend()
    plt.show()


def test():
    print("\nSelect type of fractional derivative:\n")
    print("1) Grünwald–Letnikov FD\n2) Riemann-Liouville FD\n3) Riemann-Liouville FI\n")
    print("Type response number:")
    x = input()
    if x == "1":
        return GL
    elif x == "2":
        return RL_fft
    elif x == "3":
        return RLI
    else:
        print("out of bounds")
        test()


main(RL_fft)  # Select type of FD (GL, RLI, RL and fft variants)
