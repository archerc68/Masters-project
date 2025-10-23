import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import gamma


def L1(f, alpha, x):
    h = x[1] - x[0]
    b = 1000
    x = np.arange(x[0] - b * h, x[-1] + h, h)

    dt_F_k12 = np.concatenate(([0.0], np.diff(f(x)) / h))

    j = np.arange(len(x) + 1)
    a = np.diff(j ** (1 - alpha))

    N = next_fast_len(len(dt_F_k12) + len(a) - 1)

    a_pad = np.pad(a, (0, N - len(a)))
    dt_F_k12_pad = np.pad(dt_F_k12, (0, N - len(dt_F_k12)))

    conv = irfft(rfft(a_pad) * rfft(dt_F_k12_pad), n=N)

    ans = conv * h ** (1 - alpha) / gamma(2 - alpha)

    return x[b:], ans[b : len(x)]


def L1_2(f, alpha, x):
    h = x[1] - x[0]
    b = 1000
    x = np.arange(x[0] - b * h, x[-1] + h, h)

    dt_F_k12 = np.concatenate(([0.0], np.diff(f(x)) / h))
    dt2_F_k

    j = np.arange(len(x) + 1)
    a = np.diff(j ** (1 - alpha))

    N = next_fast_len(len(dt_F_k12) + len(a) - 1)

    a_pad = np.pad(a, (0, N - len(a)))
    dt_F_k12_pad = np.pad(dt_F_k12, (0, N - len(dt_F_k12)))

    conv = irfft(rfft(a_pad) * rfft(dt_F_k12_pad), n=N)

    ans = conv * h ** (1 - alpha) / gamma(2 - alpha)

    return x[b:], ans[b : len(x)]


plt.figure()


def f(x):
    return np.exp(2*x)

alpha = 0.5
x = np.arange(0, 1, 1e-2)
x, deriv = L1(f, alpha, x)
plt.plot(x, deriv)
plt.plot(x, f(x)*2**alpha)
plt.show()
