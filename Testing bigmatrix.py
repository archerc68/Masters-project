import numpy as np
from scipy.special import gamma


# i -> i - 1
def i_shift(i, k):
    num = -(i - 1) * (i - k)
    den = i * (i + k - 1)
    return num / den


# j -> j + 1
def j_shift(j, k, nu):
    num = -j + k - nu
    den = j + k - nu + 1
    return num / den


# Lower left corner
def seed(k, nu, N):
    num = (-1) ** (N - k) * N * gamma(N + k) * gamma(k - nu + 0.5)
    den = gamma(k + 0.5) * gamma(N - k + 1) * gamma(k - nu + 1) ** 2
    return num / den


def pack(k, nu, N):
    BL = seed(k, nu, N)  # bottom left

    flatpack = np.zeros(N + 1, N + 1)

    ivals = np.arange(N+1)[:, None]
    jvals = np.arange(N+1)[None, :]

    