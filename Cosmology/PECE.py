import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from tqdm import tqdm

# mu = 2

# D^{1.5} a = f(t, a, \dot{a}) \impies q = 1/2 \implies d = 3 (min 1)
# D^{1.75} a = f(t, a, \dot{a}) \implies q = 1/4 \implies d = 7 (min 2)
# D^{1.85} a = f(t, a, \dot{a}) \implies q = 1/20 \implies d = 37 (min 10)
# D^{1.9} a = f(t, a, \dot{a}) \implies q = 1/20 \implies d = 37 (min 10)

# a(0) = a_0 (=0), \dot{a}(0) = a_dot_0

# Transformed into D^q Y(x) = g(x, Y(x))

# Y := (y_0, y_1, ..., y_{d-1})^T
# g(Y) := (y_1, y_2, ..., y_{d-1}, f(y_0, y_{\alpha1/q}), ..., y_{alpha})^T

N = int(5e3)
h = 0.5 / N


# Y(0); y_bc = [y_0, y_dot_0]
def Y_0(y_bc, d, q):
    y_0_array = np.zeros(d)
    jq = np.arange(d) * q

    mask = np.equal(jq, np.floor(jq))  # indices of rational numbers of jq
    j_bc = np.arange(d)[mask]  # j values

    y_0_array[j_bc] = y_bc
    return y_0_array


def PE(q, d, Y0, Y_h, n, g):
    j = np.arange(n + 1)
    b_jn = h**q / q * ((n + 1 - j) ** q - (n - j) ** q)  # b coefficients
    conv = np.zeros(d)
    for j in range(n + 1):
        conv += b_jn[j] * g(Y_h[j])

    y_P = Y0 + (1 / gamma(q)) * conv
    return y_P


def CE(q, d, Y0, Y_h, n, g, y_p):
    j = np.arange(n + 1)
    a_0n = n * (q + 1) - (n - q) * (n + 1) ** q  # a coefficients
    a_jn = (n - j + 2) ** (q + 1) + (n - j) ** (q + 1) - 2 * (n - j + 1) ** (q + 1)
    conv = np.zeros(d)
    conv += a_0n * g(Y_h[0])
    for j in range(1, n + 1):
        conv += a_jn[j] * g(Y_h[j])

    y_P = Y0 + h**q/gamma(q+2) * g(y_p)+(h**q / gamma(q+2)) * conv
    return y_P


def PECE(q, d, f, y_bc):
    y_h = np.zeros((N+1, d))
    Y0 = Y_0(y_bc, d, q)
    y_h[0] = Y0

    # g(x, Y)
    def g(Y):
        gvec = np.empty_like(Y)
        gvec[:-1] = Y[1:]  # Shifting y_0 -> y_1 etc.
        gvec[-1] = f(Y[0], Y[int(1 / q)])  # Last value of g(Y)
        return gvec

    for n in tqdm(range(N)):
        y_p = PE(q, d, Y0, y_h, n, g)   # Predictor
        y_h[n+1] = CE(q, d, Y0, y_h, n, g, y_p) # Corrector
    
    return y_h[:, 0]

def f(a, a_dot):
    aSoft = a + 0
    return 70 * a_dot * np.sqrt(1e-4*aSoft**(-4) + 0.2 * aSoft **(-3) + 0.8)

a = PECE(1/2, 3, f, np.array([1e-6, 1e-6]))

fig, axs = plt.subplots(1, 1)
# axs.set_xscale("log")
# axs.set_yscale("log")

i = np.linspace(0, np.log1p(len(a) - 1), 250)
indices = np.unique(np.expm1(i).astype(int))


axs.plot(np.linspace(0, 0.5, len(indices)), a[indices])
plt.show()