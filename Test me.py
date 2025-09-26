import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, gamma

limit = int(1e2)
num = 1000
alphas = np.linspace(0, 1, num)

k = np.arange(limit + 1)
a_k = np.empty((num, limit + 1))
a_k[:, 0] = 1
a_k[:, 1:] = 1 - (alphas[:, None] + 1) / k[1:]
a_k = np.cumprod(a_k, axis=1)
errors = np.sum(a_k, axis=1)


def wag(alpha):
    n1 = ((-1) ** limit) * (limit + 1)
    n2 = gamma(alpha + 1)

    d1 = alpha * gamma(limit + 2)
    d2 = gamma(alpha - limit)

    return (n1 * n2) / (d1 * d2)


plt.figure()
plt.plot(alphas, errors)
plt.plot(alphas, wag(alphas))
plt.show()
