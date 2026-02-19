import numpy as np
from numpy.polynomial.chebyshev import chebvander
import matplotlib.pyplot as plt
from scipy.fftpack import dct


# Function
def f(x):
    return x**2 - x - 1 #np.where(x > 0, 1, 0)


# Chebyshev roots
N = 5
k = np.arange(N + 1)


# Roots grid
x_k_root = -np.cos(np.pi * (2 * k + 1) / (2 * (N + 1)))
Tn_Xk_root = chebvander(x_k_root, N)

# c_n
c_n = 2 / (N + 1) * f(x_k_root) @ Tn_Xk_root 
c_n[0] *= 1 / 2

# Q_n
x = np.linspace(-1, 1, 250)
Q_n =  c_n @ chebvander(x, N).T

# Extrema grid
x_k_ext = -np.cos(k * np.pi / N)
Tn_Xk_ext = chebvander(x_k_ext, N)

# b_n
b_n = 2 / N * f(x_k_ext) @ Tn_Xk_ext
b_n[0] *= 1/2
b_n[-1] *= 1/2

# P_n
P_n =  b_n @ chebvander(x, N).T

plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))
plt.plot(x, Q_n)
plt.plot(x, P_n)
plt.plot(x, f(x))
plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))
plt.plot(x, Q_n - f(x))
plt.plot(x, P_n - f(x))
plt.show()