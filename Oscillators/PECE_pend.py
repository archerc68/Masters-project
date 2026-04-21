import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


# PECE
def PECE(alpha, bc, f, T=1, N=int(500)):
    h = T / N
    k = np.arange(1, N + 1)
    t = np.arange(0, N + 1) * h

    n = int(np.ceil(alpha))

    assert len(bc) == n

    # a and b coefficients
    b = k**alpha - (k - 1) ** alpha
    a = (k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)

    y = np.empty(N + 1)
    y_0 = bc[0]
    y[0] = y_0

    def y_bc(bc, t, n):
        if n == 1:
            return y_0
        elif n == 2:
            return y_0 + bc[1] * t

    for j in range(1, N + 1):
        YBC = y_bc(bc, t[j], n)

        # Predictor
        b_sum = 0.0
        for k in range(j):
            b_sum += b[j - k - 1] * f(t[k], y[k])
        p = YBC + h**alpha * b_sum / gamma(alpha + 1)

        # Corrector
        a_sum = 0.0
        for k in range(1, j):
            a_sum += a[j - k - 1] * f(t[k], y[k])
        a_const = (j - 1) ** (alpha + 1) - (j - 1 - alpha) * j**alpha
        y[j] = YBC + h**alpha / gamma(alpha + 2) * (
            f(t[j], p) + a_const * f(t[0], y_0) + a_sum
        )

    return t, np.real(y)


# Test case

t_0 = 50

def f(t, x):
    omega = 1
    if t < t_0:
        return -(omega**2) * x + np.sin(t)
    else:
        return -(omega**2) * x
    
def f_NA(t, x):
    omega = 1
    return -(omega**2) * x


N = 1000
T = 100
alpha = 1.9
bc = np.array([0, 0], dtype=float)

t, y = PECE(alpha, bc=bc, f=f, T=T, N=N)

transition = int(t_0/T*N)   # t_0 index
y0 = y[transition]    # y(t_0)
t_NA, y_NA = PECE(1.9, bc=np.array([y0, 0]), f=f_NA, T=T - t_0, N=N - transition) # No annealing

fig, (axs1, axs2) = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True)
plt.subplots_adjust(hspace=0)

axs1.plot(t, y, label="Annealed")
axs1.plot(t_NA+t_0, y_NA, label=r"Starts at $t_0$")

difft, diffy = t[transition:], y_NA-y[transition:]
diffmax = np.max(np.absolute(diffy))
axs2.plot(difft, diffy)

axs1.axhline(0, 0, T, linestyle="--", color="k")
axs1.axvline(t_0, -10, 10, linestyle="--", color="k")

axs2.axhline(0, 0, T, linestyle="--", color="k")
axs2.axvline(t_0, -10, 10, linestyle="--", color="k")

t_fill = np.linspace(0, t_0, 2)
y_fill = np.ones_like(t_fill)
axs2.fill_between(t_fill, -10*y_fill, 10*y_fill, alpha=0.5, color="gray")
axs2.set_ylim(-1.1*diffmax, 1.1*diffmax)

axs1.set_xlim(0, T)

axs1.set_ylabel("y")

axs2.set_xlabel("t")
axs2.set_ylabel(r"$y_{anneal} - y_{t0}$")

axs1.text(t_0/2, 1.15*np.max(y),
         "Driven",
         va="bottom",
         ha="center")

axs1.text(t_0, 1.15*np.max(y),
         r"$t_0$",
         va="bottom",
         ha="center")

axs1.text(3*t_0/2, 1.15*np.max(y),
         "Undriven",
         va="bottom",
         ha="center")

axs1.legend()

plt.savefig("Figures/CaputoDrive.svg")

plt.show()
