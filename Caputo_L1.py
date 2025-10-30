import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import gamma


def L1(f, alpha, x):
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


def bvals(j, alpha):
    diff2 = (j + 1) ** (2 - alpha) - j ** (2 - alpha)
    diff1 = (j + 1) ** (1 - alpha) - j ** (1 - alpha)
    return (diff2 / (2 - alpha)) - (diff1 / 2)


def L1_2(f, alpha, x):
    h = x[1] - x[0]
    buffer = 1000
    x = np.arange(x[0] - buffer * h, x[-1] + h, h)

    F_k = f(x)

    # GL kernals
    G1 = np.array([1, -1]) / h  # D^1
    G2 = np.array([1, -2, 1]) / (h * h)  # D^2

    j = np.arange(len(x) + 1)
    a = np.diff(j ** (1 - alpha))
    b = bvals(j, alpha)

    N = next_fast_len(len(F_k) + len(a) + len(b) + 4)

    F_k_pad = np.pad(F_k, (0, N - len(F_k)))
    a_pad = np.pad(a, (0, N - len(a)))
    b_pad = np.pad(b, (0, N - len(b)))
    G1_pad = np.pad(G1, (0, N - 2))
    G2_pad = np.pad(G2, (0, N - 3))
    
    L_1 = rfft(a_pad) * rfft(G1_pad)
    correction = h * rfft(b_pad) * rfft(G2_pad)

    conv = irfft((L_1 + correction) * rfft(F_k_pad))

    ans = conv * h ** (1 - alpha) / gamma(2 - alpha)

    return x[buffer:], ans[buffer : len(x)]


plt.figure()


def f(x):
    return np.cos(x)


alpha = 0.5
xin = np.arange(0, 2 * np.pi, 1e-2)
_, L1_FD = L1(f, alpha, xin)
plt.plot(xin, L1_FD, label="L1")
_, L12_FD = L1_2(f, alpha, xin)
plt.plot(xin, L12_FD, label="L1-2")
plt.plot(xin, f(xin + alpha * np.pi / 2), label="Correct")
plt.legend()
print(np.sum(L12_FD - L1_FD))
plt.show()


