import numpy as np
from scipy.special import gamma, gamma, poch, rgamma, gammaln, loggamma
from gmpy2 import mpfr

L = 1


# i -> i - 1
def I(i, k):
    num = -(i - 1) * (i - k)
    den = i * (i + k - 1)
    return num / den


# j -> j + 1
def J(j, k, nu):
    num = -j + k - nu
    den = j + k - nu + 1
    return num / den



def seeds(nu, N):
    n = int(np.ceil(nu))

    s1 = (-1) ** (N - n)
    s2 = gamma(N + n) * rgamma(N - n + 1)
    s3 = gamma(n - nu + 0.5) * rgamma(n + 0.5)
    s4 = N * rgamma(n - nu + 1) ** 2

    seed0 = s1 * s2 * s3 * s4

    seed_arr = np.zeros(N - n + 1)
    seed_arr[0] = 1

    for k in range(n, N):
        num = -(N + k) * (k - nu + 0.5) * (N - k)
        den = (k + 0.5) * (k - nu + 1) ** 2
        seed_arr[k - n + 1] = num / den * seed_arr[k - n]

    return seed0 * seed_arr


def D(N, nu):
    n = int(np.ceil(nu))
    BigMat = np.zeros((N + 1, N + 1))
    c = np.zeros_like(BigMat)
    seed_arr = seeds(nu, N)

    jvals = np.arange(N + 1)

    for k in range(n, N + 1):
        SubMat = np.zeros((N + 1, N + 1))
        SubMat[N, 0] = seed_arr[k - n]

        JOp = J(jvals[:N], k, nu)
        SubMat[N, 1:] = SubMat[N, 0] * np.cumprod(JOp)

        Is = np.arange(k + 1, N + 1)[::-1]

        for i in Is:
            SubMat[i - 1, :] = I(i, k) * SubMat[i, :]

        y = SubMat - c
        t = BigMat + y
        c = (t - BigMat) - y
        BigMat = t.copy()

    eps_j = np.ones((N + 1, N + 1))
    eps_j[:, 0] = 2

    coeff = 2 / (eps_j * L**nu)

    return coeff * BigMat


def D1(N, nu):
    LB = int(np.ceil(nu))

    i = np.arange(N + 1, dtype=int)[:, None]
    j = np.arange(N + 1, dtype=int)[None, :]
    D_matrix = np.zeros((N + 1, N + 1))

    eps_j = np.ones((N + 1, N + 1))
    eps_j[:, 0] = 2

    coeff = 2 * i / (eps_j * L**nu)
    sign = np.where((i - LB) % 2 == 0, 1, -1)

    for k in range(LB, N + 1):
        a = k - nu + 1

        # Numerator & denominator
        log_num = loggamma(i + k) + loggamma(k - nu + 0.5)
        log_den = loggamma(k + 0.5) + gammaln(i - k + 1) + 2 * loggamma(a)

        num_den = np.exp(log_num - log_den)

        # Corrective terms to allow logarithms
        # [loggamma(k - j - nu + 1) woud return errors]
        # Terms derived from gamma(a + j) * gamma(a - j)

        factors = (a + j - 1) / (a - j)
        factors[:, 0] = 1
        term = np.cumprod(factors, axis=1)

        iteration = num_den * coeff * sign / term
        sign *= -1

        # Masking values
        iteration = np.where(k <= i, iteration, 0)
        D_matrix += iteration

    return D_matrix


