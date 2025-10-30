import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.fft import rfft, irfft, next_fast_len

# def Caputo(alpha, x, y):
#     h = x[1] - x[0]
#     b = 2

#     # Kernal
#     factors = 1 - (alpha + 1) / np.arange(1, len(x) - 1)
#     w = np.concatenate(([1.0], np.cumprod(factors)))
#     diff = np.diff(y)

#     # Padding
#     N = len(diff) + len(w) - 1
#     w_pad = np.pad(w, (0, N - len(w)))
#     f_pad = np.pad(diff, (0, N - len(y)))

#     w_hat = rfft(w_pad)
#     f_hat = rfft(f_pad)

#     conv = irfft(w_hat*f_hat)[:len(x)]

#     ans = conv / (h**alpha * gamma(2 - alpha))

#     return x[b:], ans[b:]



def Caputo(alpha, x, y, b):
    """
    Caputo fractional derivative (0<alpha<1) using WL1 scheme
    with FFT-based linear convolution (explicit zero padding).
    """
    h = x[1] - x[0]
    y = np.asarray(y, dtype=float)
    N = y.size
    b = 250
    if N < 2:
        return np.zeros_like(y)

    # Forward differences Δy (length M)
    delta_y = np.diff(y)
    M = delta_y.size

    # WL1 weights w_j^(alpha), j=0..M-1 (vectorised)
    if M > 0:
        factors = 1 - (alpha + 1) / np.arange(1, M)
        w = np.concatenate(([1.0], np.cumprod(factors)))
    else:
        w = np.array([], dtype=float)

    # Linear convolution length
    L = M + M - 1
    nfft = next_fast_len(L)

    # Explicit zero padding to nfft
    delta_padded = np.pad(delta_y, (0, nfft - M), mode='constant')
    w_padded     = np.pad(w,       (0, nfft - M), mode='constant')

    # FFT-based linear convolution
    F_delta = rfft(delta_padded)
    F_w     = rfft(w_padded)
    conv    = irfft(F_delta * F_w, n=nfft)[:L]

    # Extract sums for n = 1..N-1 (indices 0..M-1)
    sums = conv[:M]

    D = np.zeros(N, dtype=float)
    D[1:] = sums / (gamma(2 - alpha)*h**(alpha + 1))
    return x[b:], D[b:]


plt.figure()
b = 250
h = 1e-2
x = np.arange(0-b*h, 4*np.pi, h)
y = np.cos(x)
x, D = Caputo(0.5, x, y, 250)
plt.plot(x, D)
plt.show()
