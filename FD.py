import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


# Grünwald–Letnikov FD alpha \in [0, inf)
def GL(f, alpha, x_min, x_max):
    # Parameters
    steps = 250
    h = 1e-2

    # Kernal
    k = np.arange(steps)
    kernal = np.empty(steps)
    kernal[0] = 1
    kernal[1:] = 1 - (alpha + 1) / k[1:]
    kernal = np.cumprod(kernal) / h**alpha

    # f(x)
    x = np.arange(x_min - h * steps, x_max + h, h)
    f_x = f(x)

    # Discrete convolution
    FD = fftconvolve(kernal, f_x, "valid")

    return x[steps - 1 :], FD


# Plotting
def main():
    def f(x):
        return np.exp(2 * x)

    plt.figure()
    num = 50
    for i in range(num):
        x, FD = GL(f, 2 * i / num, 0, 2 * np.pi)
        plt.plot(x, FD)
    plt.show()

main()
