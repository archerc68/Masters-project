import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import gamma


def L1(f, alpha, x):
    h = x[1] - x[0]
    b = 250
    x = np.arange(x[0] - b * h, x[-1] + h, h)

    f_diff = np.empty_like(x)
    f_diff[0] = 0
    y = f(x)
    for i in range(1, len(x)):
        f_diff[i] = (y[i] - y[i - 1]) / h
    print(f_diff[:5])

    j = np.arange(len(x))
    a = np.diff(j ** (1 - alpha))

    N = next_fast_len(len(f_diff) + len(a) - 1)

    a_pad = np.pad(a, (0, N - len(a)))
    f_diff_pad = np.pad(f_diff, (0, N - len(f_diff)))

    conv = irfft(rfft(a_pad) * rfft(f_diff_pad))

    ans = conv * h ** (1 - alpha) / gamma(2 - alpha)

    return x[b:], ans[b : len(x)]


plt.figure()


def f(x):
    return np.cos(x)


x = np.arange(0, 2 * np.pi, 1e-2)
x, deriv = L1(f, 0.1, x)
plt.plot(x, deriv)
plt.show()
