import numpy as np
from math import gamma, factorial
import matplotlib.pyplot as plt

def choses(n, x):
    def gammas(x):
        return np.array([gamma(n + 1) for n in x])

    return gamma(n + 1) / (gammas(n - x) * gammas(x))

def deriv(f, N, x):

    h = 1e-2

    sign = np.empty((N + 1))
    sign[::2] = -1
    sign[1::2] = 1

    fx = f(np.linspace(x, x + N * h, N + 1))

    coeff = sign * choses(N, np.arange(0, N + 1)) * fx

    return np.sum(coeff) / (h**N)


# def f(x):
#     return np.array([gamma(i) for i in x])


#print(deriv(f, 1, 4))

def chose(a, b):
    if b > a:
        return 0
    else:
        return gamma(a + 1)/(gamma(b + 1)*gamma(a - b + 1))


def Leibenez(f, g, alpha, D, t):
    limit = 100 # Resolution

    result = 0
    for i in range(limit):
        result += chose(alpha, i)*D(f, i, t)*D(g, alpha - i, t)

    return result

print(Leibenez(f = lambda x: 2*x + 1, g = lambda x: x, alpha = 2, D = deriv, t = 0))


