import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

t = np.linspace(-1, 1, 40)
dt = t[1] - t[0]

def f(x):
    return np.tanh(2*x)

y = f(t)


fig, axs = plt.subplots(1, 1)

axs.set_xlim(t[0], t[-1])
axs.set_ylim(-1, 1)


def GL_kernal(alpha):
    h = dt
    p = int(len(t)/2)
    k = np.arange(p + 1)
    tk =  - h* k

    chose = gamma(alpha + 1)/(gamma(alpha - k + 1)*gamma(k + 1))

    return tk, (-1) ** k * chose/2**alpha

k1, GL1 = GL_kernal(0.5)
k2, GL2 = GL_kernal(1)
axs.bar(k1+dt/2, GL1, width=dt, edgecolor="r", facecolor = "white", alpha=0.4, label=r"$\alpha = 0.5$", hatch="//")
axs.bar(k2+dt/2, GL2, width=dt, edgecolor="g", facecolor = "g", alpha=0.4, label=r"$\alpha = 1$")

print(2**0.5*GL1)

talt = np.linspace(t[0], t[-1], 250)
axs.plot(talt, f(talt), color="b")
axs.scatter(t, f(t), color="b", label=r"$y=\tanh(2 x)$")

axs.legend()

axs.set_xlabel(r"$x$")
axs.set_ylabel(r"$y$")
axs.set_title("Normalised kernals")

plt.savefig("FFT/GL_kernal_tanh.png", dpi=250)
plt.show()


