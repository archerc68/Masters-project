import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import fftconvolve
from scipy.special import gamma
from alive_progress import alive_bar


### Kernals ##
def GLkernal(n_max, h, alpha):
    k = np.arange(n_max)
    g_k = np.ones(n_max)
    g_k[1:] -= (alpha + 1) / k[1:]
    return np.cumprod(g_k)/(h**alpha)


def RLkernal(n, h, alpha):
    k = np.arange(0, n)
    return ((h**alpha) / gamma(alpha)) * k ** (alpha - 1)


### GL ###
def GL(f, alpha, x, n=1000):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    G_k = GLkernal(n, h, alpha)
    F_j = f(x)
    return x[b:], fftconvolve(F_j, G_k)[b : len(x)]


### Riemann-Liouville ###


# RLI -- currently broken
def RLI(f, alpha, x):
    h = x[1] - x[0]
    F_k = f(x)
    R_k = RLkernal(len(x), h, alpha)
    conv = fftconvolve(F_k, R_k)[: len(x)]

    conv -= 0.5 * F_k[0] * R_k[0]
    return x, conv


# RL
def RL(f, alpha, x):
    def g(x):
        x, y = RLI(f, np.ceil(alpha) - alpha + 1, x)
        return y

    x, deriv = GL(g, np.ceil(alpha) + 1, x, n=int(np.ceil(alpha) + 2))
    return x, deriv


# RLI using ffts -- currently broken
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


# RL using ffts
def RL_fft(f, alpha, x):
    # Parameters
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    n = int(np.ceil(alpha) + 1)

    # Kernals
    k = np.arange(len(x))

    # G'_k
    G_k = np.ones(n + 1)  # Of length n + 1 since n is an integar
    G_k[1:] -= (n + 1) / k[1 : (n + 1)]
    G_k = np.cumprod(G_k)

    # R'_k
    R_k = k ** (n - alpha - 1)

    # F_j
    F_j = f(x)

    # Linear convolution
    N = 2 * len(x) + n
    Rpad = np.pad(R_k, (0, N - len(x)))
    Gpad = np.pad(G_k, (0, N - n - 1))
    Fpad = np.pad(F_j, (0, N - len(x)))

    RG = rfft(Rpad) * rfft(Gpad)
    FRG = rfft(Fpad) * RG
    correction = 0.5 * F_j[0] * RG

    conv = irfft(FRG - correction)
    ans = conv / (gamma(n - alpha) * (h**alpha))

    return x[b:], ans[b : len(x)]


### Plotting ###


def main(FD):
    # Alpha values
    num = 5000
    alphas = np.linspace(0, 1, num)

    # Function
    def f(x):
        return np.exp(-x*x)

    # Displaying plot
    plt.figure()
    with alive_bar(num) as bar:
        for i in range(len(alphas)):
            x = np.arange(
                -2 * np.pi, 2 * np.pi, 1e-2
            )  # Increasing resolution may warrant an increase in GL buffer size
            x, y = FD(f, float(alphas[i]), x)
            bar()

            plt.plot(x, y, label=str(alphas[i]))
    print("Plotting...")
    # plt.legend()
    #plt.show()
    print("Done")


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
