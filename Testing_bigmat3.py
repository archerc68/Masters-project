import numpy as np
from scipy.special import gamma, loggamma, gammaln
import matplotlib.pyplot as plt

L = 2


def I_plus(i, n):
    def I_plusn(i):
        num = -i * (i + n - 1)
        den = (i - 1) * (i - n)
        return num/den
    return np.where(i > n, I_plusn(i), 1)


def J_plus(j, nu, n):
    num = (-j + n - nu + 1)
    den = (j + n - nu)

    ansJ = num/den
    ansJ[j == 0] = 1
    return ansJ


def K_plus(i, j, k, nu):
    def k_plusi(k):
        num = (i * i - (k - 1) ** 2) * (2 * k - 2 * nu - 1)
        den = (j * j - (k - nu) ** 2) * (2 * k - 1)
        return num / den

    ansK = np.where(k <= i, k_plusi(k), 0)
    ansK[:, :, 0] = 1

    return ansK

def seed(nu, n):
    num = 2 ** (2 * n - 1) * gamma(n + 1) * gamma(-nu + n + 0.5)
    den = np.sqrt(np.pi) * gamma(-nu + n + 1) ** 2
    return num / den


def BigMat(N, nu):
    if type(nu) is int:
        return print("yes")
    else:
        n = int(np.ceil(nu))
        Mat = np.zeros((N + 1, N + 1))

        arrij = np.arange(N + 1)
        arrk = np.arange(N + 1 - n) + n

        I, J = I_plus(arrk, n), J_plus(arrij, nu, n)

        i = arrij[:, None, None]
        j = arrij[None, :, None]
        k = arrk[None, None, :]

        K = K_plus(i, j, k, nu)

        Mat[n:, 0] = seed(nu, n) * np.cumprod(I)

        Mat = Mat[:, 0][:, None] * np.cumprod(J)[None, :]

        Mat = Mat * np.sum(np.cumprod(K, axis=2), axis=2)

        # Prefactors
        eps_j = np.ones((N+1, N+1))
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

        for k in range(LB, N+1):
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


print(BigMat(20, 1.85)/D1(20, 1.85))
# BigMat(5, 1.85)
