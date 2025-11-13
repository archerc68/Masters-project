import numpy as np
from scipy.special import gamma, loggamma, rgamma, factorial

a, n = 3, 2

x = gamma(a + n) * gamma(a - n)
print(x)

y = gamma(a) ** 2 * np.prod([(a + k) / (a - n + k) for k in range(0, n)])
print(y)

print("logs:")

# Logs
# x_log = loggamma(a+n) + loggamma(a-n)
# print(x_log)

# y_log = 2*loggamma(a) + np.sum([np.log(a + k) - np.log(a - n + k) for k in range(0, n)])
# print(y_log)

# Hybrid
print("Hybrid:")
y_hybrid = np.exp(2 * loggamma(a)) * np.prod(
    [(a + k) / (a - n + k) for k in range(0, n)]
)
print(y_hybrid)


def D(N, nu):
    LB = int(np.ceil(nu))

    i = np.arange(N + 1)[:, None]
    j = np.arange(N + 1)[None, :]
    D_matrix = np.zeros((N + 1, N + 1))

    eps_j = np.ones_like(j)
    eps_j[:, 0] = 2

    for k in range(LB, N + 1):
        sign = np.where((i - k) % 2 == 0, 1, -1)
        log_num = loggamma(i + k) + loggamma(k - nu + 0.5)
        log_den = loggamma(k + 0.5) + loggamma(i - k + 1) + 2 * loggamma(k - nu + 1)

        num_den = np.exp(log_num - log_den)

        coeff = 2 * i * sign /(eps_j * L**nu)

        a = k - nu + 1
        factors = (a + j - 1) / (a - j)
        factors[:, 0] = 1
        term = np.cumprod(factors, axis=1)

        iteration = num_den * coeff/term

        iteration = np.where(k <= i, iteration, 0)
        D_matrix += iteration

    return D_matrix

print(D(N=5, nu=1.5))


# [[ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.25801228  0.17200818 -0.03440164  0.01474356 -0.00819087  0.00521237]
#  [-0.17200818  0.61922946  0.44230676 -0.11467212  0.05629359 -0.0340236 ]
#  [ 1.19717696  0.29487117  0.77977043  0.8095554  -0.24446584  0.13093471]
#  [-0.68452236  2.22463918  0.82653283  0.92619791  1.26973314 -0.4258509 ]]

# j = np.arange(5)[None, :]
# i = np.zeros((5, 5))
# j = i + j

# # print(j)

# a, n = 3.7, 2

# factors = (a + j - 1) / (a - j)
# factors[:, 0] = 1
# new = gamma(a) ** 2 * np.cumprod(factors, axis=1)

# print(new)

# old = gamma(a + j) * gamma(a - j)
# print(old)

# print(new - old)
