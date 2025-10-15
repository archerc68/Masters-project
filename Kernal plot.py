import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft
from scipy.special import gamma


def GLkernal(n_max, h, alpha):
    k = np.arange(n_max + 1)
    g_k = np.empty(n_max + 1)
    g_k[0] = 1 / (h**alpha)
    g_k[1:] = 1 - (alpha + 1) / k[1:]

    return np.cumprod(g_k)


def GL(f, alpha, x):
    n = 1000
    h = x[1] - x[0]
    x = np.arange(x[0] -n*h, x[-1] + h)
    F_j = f(x)
    G_k = GLkernal(n, h, alpha)
    y = fftconvolve(F_j, G_k)[:len(x)]
    
    return x[n:], y[n:]


def RLkernal(n, h, alpha):
    k = np.arange(n+1)
    return ((h**alpha)/gamma(alpha))*k**(alpha - 1)


def RLI(f, alpha, x):
    h = x[1] - x[0]
    F_k = f(x)
    R_k = RLkernal(len(x)-1, h, alpha)
    conv = fftconvolve(F_k, R_k)[:len(x)]

    conv -= 0.5*F_k[0]*R_k[0]
    return x, conv

def RL(f, alpha, x):
    def g(x):
        x, y = RLI(f, np.ceil(alpha) - alpha + 1, x)
        return y
    x, deriv = GL(g, np.ceil(alpha) + 1, x)
    return x, deriv

def RLA(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250 # Buffer
    x = np.arange(x[0]-b*h, x[-1]+h, h)
    #Kernals
    R_k = RLkernal(len(x)-1, h, np.ceil(alpha) - alpha + 1)
    G_k = GLkernal(len(x)-1, h, np.ceil(alpha) + 1)
    F_j = f(x)
    # RL
    temp1 = fft(R_k)*fft(G_k)
    temp2 = (fft(F_j) - 0.5*F_j[0])*temp1
    return x[b:], np.real(ifft(temp2)[b:])

def GLA(f, alpha, x):
    # Params
    h = x[1] - x[0]
    b = 250 # Buffer
    x = np.arange(x[0]-b*h, x[-1]+h, h)
    #Kernals
    G_k = GLkernal(len(x)-1, h, alpha)
    F_j = f(x)
    # GL
    temp1 = fft(G_k)*fft(F_j)
    return x[b:], np.real(ifft(temp1)[b:])


alphas = np.linspace(1, 2, 5)

def f(x):
    return np.cos(x)

plt.figure()
for i in range(len(alphas)):
    x = np.arange(0, 2*np.pi, 1e-2)
    x, y = RLA(f, float(alphas[i]), x)
    plt.plot(x, y, label=str(alphas[i]))
plt.legend()
plt.show()
