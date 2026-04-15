import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


def polyderiv(n, m, x):
    return gamma(n + 1) / gamma(n - m + 1) * x ** (n - m)


fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 2, 250)

alphas = np.linspace(0, 2, 7)
cmap = plt.get_cmap("plasma_r", len(alphas))

norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(
    sm,
    ticks=alphas,
    ax=ax,
    label=r"$\alpha$",
)

for i in range(len(alphas)):
    alpha = alphas[i]
    if np.isclose(alpha, np.round(alpha), rtol=1e-2):
        ax.plot(x, polyderiv(3, alpha, x), linestyle="--", color=cmap(i))
    else:
        ax.plot(x, polyderiv(3, alpha, x), color=cmap(i))

x1, y1 = 1.05, polyderiv(3, 0, 1.05)
x2, y2 = 1.15, polyderiv(3, 2, 1.15)

ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="<-", color="black",
                            shrinkA=5, shrinkB=5, lw=1,
                            patchA=None, patchB=None,
                            connectionstyle="arc3,rad=-0.25",
                            ),
            )


ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$D^{\alpha}x^3$")

plt.savefig("Figures/Polyinterp.png", dpi=250)
plt.show()
