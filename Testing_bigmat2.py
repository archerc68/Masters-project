import numpy as np
from scipy.special import poch, rgamma, loggamma, gammaln

L = 1


def D_1(N):
    D_matrix_T = np.zeros((N + 1, N + 1))
    k = np.arange(1, N + 1, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4 / L
    return np.array(D_matrix)


# i + 1 -> i
def I(i, k, N):
    num = -i * (i - k + 1)
    den = (i + 1) * (i + k)
    frac = num / den
    return np.where(i == N, 1, frac)


# j - 1 -> j
def J(j, k, nu):
    num = -j + k - nu + 1
    den = j + k - nu
    frac = num / den
    frac[:, 0] = 1
    return frac


# k - 1 -> k
def K2(i, j, k, N, nu):  # Needs fixing

    num = (i * i - (k - 1) ** 2) * (2 * k - 2 * nu - 1)
    den = (j * j - (k - nu) ** 2) * (2 * k - 1)
    frac = num / den

    frac[:, :, 0] = 1

    print("\nFrac:\n")
    print(frac)

    rec = np.cumprod(frac, axis=2)

    print("\nRec:\n")
    print(rec)

    return np.sum(rec, axis=2)


def K(i, j, k, nu):
    
    num = (i * i - (k - 1) ** 2) * (2 * k - 2 * nu - 1)
    den = (j * j - (k - nu) ** 2) * (2 * k - 1)
    frac = num / den

    ratio = np.where(k >= i, frac, 0)
    ratio[..., 0] = 1

    print(ratio)

    ratio = np.cumprod(ratio, axis=2)

    print(ratio)

    return np.sum(ratio, axis=2)


def seed_0(N, nu):
    n = int(np.ceil(nu))
    num = (-1) ** (N - n) * N * poch(n + 0.5, -nu) * poch(N - n + 1, 2 * n - 1)
    den_inv = rgamma(n - nu + 1) ** 2
    return num * den_inv


def BigMat(N, nu):
    if type(nu) is int:
        return np.linalg.matrix_power(D_1(N), nu)
    else:
        n = int(np.ceil(nu))

        arrij = np.arange(N + 1, dtype=int)
        arrk = np.arange(N - n + 1, dtype=int) + n

        i = arrij[:, None, None]
        j = arrij[None, :, None]
        k = arrk[None, None, :]

        # Ratios of consecutive terms
        Si, Sj = I(arrij, n, N), J(arrij[None, :], n, nu)
        Sk = K(i, j, k, nu)

        # i recurrence relations
        Mat_i = np.cumprod(Si[::-1])[::-1] * seed_0(N, nu)

        # i recurrence relations
        Mat_ij = np.cumprod(Sj, axis=1) * Mat_i[:, None]

        # k recurrence relations
        Mat = Mat_ij * Sk

        # Prefactors
        eps_j = np.ones_like(Mat_ij)
        eps_j[:, 0] = 2
        coeff = 2 / (L ** nu * eps_j)

        return coeff * Mat


def D1(N, nu):
    if type(nu) is int:
        return np.linalg.matrix_power(D_1(N), nu)
    else:
        LB = int(np.ceil(nu))

        i = np.arange(N + 1, dtype=int)[:, None]
        j = np.arange(N + 1, dtype=int)[None, :]
        D_matrix = np.zeros((N + 1, N + 1))

        eps_j = np.ones((N + 1, N + 1))
        eps_j[:, 0] = 2

        coeff = 2 * i / (eps_j * L**nu)
        sign = np.where((i - LB) % 2 == 0, 1, -1)

        for k in range(LB, LB+1):
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


# print(np.max(np.abs(BigMat(50, 1.5) - D1(50, 1.5))))


# N = 50
# i, j, k = np.meshgrid(np.array([N]), np.array([0]), 2 + np.arange(N + 1 - 2) + 2)
# print(seed_0(5, 1.85)*K(i, j, k, 1.85))

print(D1(5, 1.85))
