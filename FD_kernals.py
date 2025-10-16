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
    k = np.arange(n)
    return ((h**alpha) / gamma(alpha)) * k ** (alpha - 1)


### GL ###
def GL(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    G_k = GLkernal(len(x), h, alpha)
    F_j = f(x)
    # GL
    temp1 = fft(G_k) * fft(F_j)
    return x[b:], np.real(ifft(temp1)[b:])


### Riemann-Liouville ###


# RLI
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

    x, deriv = GL(g, np.ceil(alpha) + 1, x)
    return x, deriv


# RLI using ffts
def RLI_fft(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    R_k = RLkernal(len(x), h, alpha)
    F_j = f(x)
    # RL
    temp1 = fftconvolve(R_k, F_j)[: len(x)]
    temp2 = temp1 - 0.5 * F_j[0] * R_k
    return x[b:], np.real(temp2[b:])


# RL using ffts -- currently broken
def RL_fft(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250  # Buffer
    x = np.arange(x[0] - b * h, x[-1] + h, h)
    # Kernals
    R_k = RLkernal(len(x), h, np.ceil(alpha) - alpha + 1)
    G_k = GLkernal(len(x), h, np.ceil(alpha) + 1)
    F_j = f(x)
    # RL
    temp1 = fft(R_k) * fft(G_k)
    temp2 = (fft(F_j) - 0.5 * F_j[0]) * temp1
    return x[b:], np.real(ifft(temp2)[b:])


### Plotting ###


def main(FD):
    # Alpha values
    alphas = np.linspace(1, 2, 5)

    # Function
    def f(x):
        return np.cos(x)

    # Displaying plot
    plt.figure()
    for i in range(len(alphas)):
        x = np.arange(
            0, 2 * np.pi, 1e-2
        )  # Increasing resolution may warrant an increase in GL buffer size
        x, y = FD(f, float(alphas[i]), x)

        plt.plot(x, y, label=str(alphas[i]))
    plt.legend()
    plt.show()

def test():
    print("Select type of fractional derivative:")
    print("1) GL \n2) RLI \n3) RLI")
    x = input()
    if x == "1":
        return GL
    elif x == "2":
        return RL
    elif x == "3":
        return RLI
    else:
        print("out of bounds")
        test()

main(test())  # Select type of FD (GL, RLI, RL and fft variants)
