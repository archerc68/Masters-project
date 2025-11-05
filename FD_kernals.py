import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve
from scipy.special import gamma


### Kernals ###
def GLkernal(n_max, h, alpha):
    k = np.arange(1, n_max)
    g_k = np.ones(n_max)
    g_k[1:] -= (alpha + 1) / k
    return np.cumprod(g_k) / (h**alpha)


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


# RLI -- currently slightly broken
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


# RLI using ffts -- currently very broken
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
    G_k = np.cumprod(np.concatenate(([1.0], 1 - (n + 1) / k[1 : n + 1])))
    R_k = k ** (n - alpha - 1)
    F_j = f(x)

    # Linear convolution
    N = next_fast_len(2 * len(x) + n)
    Rpad, Gpad, Fpad = [np.pad(arr, (0, N - len(arr))) for arr in (R_k, G_k, F_j)]
    RG = rfft(Rpad) * rfft(Gpad)
    conv = irfft((rfft(Fpad) - 0.5 * F_j[0]) * RG)

    return x[b:], conv[b : len(x)] / (gamma(n - alpha) * (h**alpha))


### Caputo ###
def Caputo_L1(f, alpha, x):
    h = x[1] - x[0]
    buffer = 1000
    x = np.arange(x[0] - buffer * h, x[-1] + h, h)

    G1_k = np.array([1, -1]) / h
    F_k = f(x)

    j = np.arange(len(x) + 1)
    a = np.diff(j ** (1 - alpha))

    # Optimal padding to convort circular convolution to linear
    N = next_fast_len(len(x) + len(a) + 1)

    # Constant kernals (only depend on alpha)
    a_pad = np.pad(a, (0, N - len(a)))
    G1_k_pad = np.pad(G1_k, (0, N - 2))

    # Function output
    F_k_pad = np.pad(F_k, (0, N - len(F_k)))

    conv = irfft(
        rfft(a_pad) * rfft(G1_k_pad) * rfft(F_k_pad), n=N
    )  # Double convolution

    ans = conv * h ** (1 - alpha) / gamma(2 - alpha)

    return x[buffer:], ans[buffer : len(x)]


### Plotting ###


### Frontend ###
def test():
    options = {"1": GL, "2": RL_fft, "3": RLI, "4": Caputo_L1}

    while True:
        print("\nSelect type of fractional derivative:\n")
        print("1) Grünwald–Letnikov FD")
        print("2) Riemann-Liouville FD")
        print("3) Riemann-Liouville FI")
        print("4) Caputo FD\n")
        x = input("Type response number: ")

        if x in options:
            return options[x]
        else:
            print("\nOut of bounds, please try again.")


if __name__ == "__main__":
    FD = test()
    # Alpha values
    num = 50
    alphas = np.linspace(0.01, 0.99, num)

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
    print("Plotting...")
    plt.title(
        "Fractional derivatives between " + str(alphas[0]) + " - " + str(alphas[-1])
    )
    plt.show()
    print("Done")
