import numpy as np
from scipy.integrate import quad

k_r, k_m, k_l = 1e-4, 0.2, 0.8

a = np.linspace(0, 1, 250)
def f(a):
    return np.sqrt(k_r*a**(-4) + k_m*a**(-3)+k_l)

integral = quad(f, 1e-2, 1)
print(integral)