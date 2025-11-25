import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# PDE
# dy/dt = dy/dx

L = 10
N = 20
t = np.linspace(-1, 1, N + 1)
x = (t + 1)/2
y0 = x**2
T = 1000
dt = 0.01

def D(N, nu):
    if type(nu) is int:
        def D_1(N):
            D_matrix_T = np.zeros((N + 1, N + 1))
            k = np.arange(1, N + 1, 2)

            for i in k:
                D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
            D_matrix = D_matrix_T.T
            D_matrix[:, 0] /= 2

            D_matrix *= 4 / L
            return np.array(D_matrix)
        return np.linalg.matrix_power(D_1(N), nu)

phi = chebvander(t, N).T

i = np.arange(N+1)[:, None]
j = np.arange(N+1)[None, :]
phi = np.cos(np.pi*i*j/N)

C_T = least_squares(lambda C_T: C_T @ phi - y0, np.random.random(N+1)).x

dy_dx = D(N, 1) 
y = y0[:]
ys = np.empty((len(x), T + 1))
ys[:, 0] = y

for i in range(T):
    C_T += dt * C_T @ dy_dx @ phi
    ys[:, i] = C_T @ phi

plt.figure()
for i in range(T+1):
    plt.plot(x, ys[:, i], color="black", alpha=0.1)
plt.show()