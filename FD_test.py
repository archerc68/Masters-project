import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gamma


# Grünwald–Letnikov FD alpha \in [0, inf)
def GL(f, alpha, x_min, x_max):
    # Parameters
    steps = int(1e3)
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


def RLI(f, alpha, x_min, x_max):
    # Parameters
    dt = 1e-2
    #buffer = 10 
    a = 0

    # Minimum values used in convolution
    # Hence a = x_min - buffer*dt

    x = np.arange(a, x_max + dt, dt)
    f_x = f(x)

    # Kernal
    n = len(x)
    u = np.arange(0, (n + 1)*dt, dt)
    g_u = dt*(u**(alpha - 1))

    FD = fftconvolve(f_x, g_u, "full")[:n]

    # Correction using trapezium rule 
    FD -= dt*f_x[0]*g_u[:n]/2

    # Truncating output to match bounds
    loc = np.floor(n*(x_min - a)/(x_max - a))
    loc= int(loc)

    return x[loc:], FD[loc:]/gamma(alpha)



# Plotting
def main():
    def f(x):
        return np.sin(x)*np.exp(-x)

    plt.figure()
    num = 50
    for i in range(num):
        x, FD = RLI(f, 1 + 2 * i / num, 0, 2*np.pi)
        plt.plot(x, FD)
    plt.show()

main()
