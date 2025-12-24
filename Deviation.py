import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

plt.rcParams["font.family"] = "Times New Roman"


# Plots
plotA = False
plotB = True
plotC = False


a_2s = np.linspace(-2, 2, 250)


def trunc(a_2, x):
    return 2 * a_2 * x**2 + 1


if plotA:

    def var(a_2):
        diff_squared = lambda x: (trunc(a_2, x) - np.cos(x)) ** 2
        return np.sqrt(quad(diff_squared, -1, 1)[0])

    vars = np.empty_like(a_2s)
    for i in range(len(a_2s)):
        vars[i] = var(a_2s[i])

    min_var = np.min(vars)
    print(min_var)

    mask = np.where(vars == min_var, 1, 0)
    min_a_2 = np.dot(mask, a_2s)
    print(min_a_2)

    plt.figure()
    plt.plot(a_2s, vars)
    plt.scatter(min_a_2, min_var, c="r", label="Minimum RMS Error")
    plt.xlabel(r"$a_2$")
    plt.ylabel(r"$\sqrt{\int_{-1}^1 | u_3(x; a_2) - \cos{(x)} |^2dx}$")
    plt.title(r"$u_3(x; a_2)$ RMS Error")
    plt.legend()
    plt.savefig("u_3_error.png", dpi=1000)
    plt.show()

if plotB:

    def remainder(a_2, x):
        return 2 * (a_2 + 1) * x**2 + (4 * a_2 + 3)

    def remainder_stats(a_2):
        remainder_x = lambda x: x * remainder(a_2, x)
        mean = quad(remainder_x, -1, 1)[0]

        remainder_dev = lambda x: (x - mean) ** 2 * remainder(a_2, x)
        std = np.sqrt(np.abs(quad(remainder_dev, -1, 1)[0]))

        return mean, std

    means, stds = np.empty_like(a_2s), np.empty_like(a_2s)
    for i in range(len(a_2s)):
        means[i], stds[i] = remainder_stats(a_2s[i])

    plt.figure()

    plt.fill_between(
        a_2s,
        means - stds,
        means + stds,
        alpha=0.5,
        hatch="/",
        color="gray",
        label="Error",
    )
    plt.plot(a_2s, means, label="Mean")
    plt.scatter([-21 / 26], [0], c="r", label="Minimum error")
    plt.legend()
    plt.xlabel(r"$a_2$")
    plt.ylabel(r"$\bar{R}(a_2)$")
    plt.title("Remainder error")
    plt.savefig("remainder minimisation.png", dpi=1000)
    plt.show()


if plotC:
    x = np.linspace(-1, 1, 250)
    plt.figure()
    plt.plot(x, np.cos(x), label=r"$\cos{(x)}$")

    def Taylor(x):
        return 1 - x**2 / 2

    plt.plot(x, Taylor(x), label=r"$1-x^2/2$")
    plt.plot(x, trunc(-21 / 26, x), label=r"$u_3(x;a_2=-21/26)$")
    plt.plot(x, trunc(-62 / 99, x), label=r"$u_3(x;a_2=-62/99)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Analytic and numerical solutions for $\frac{d^2y}{dx^2} + y = 0$")
    plt.legend()
    plt.savefig("Approximations.png", dpi=1000)
    plt.show()
