import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

plt.rcParams["font.family"] = "Times New Roman"


# Plots
plotA = False
plotB = False
plotC = False
plotD = True

Save_figure = True


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

    mask = np.where(vars == min_var, 1, 0)
    min_a_2 = np.dot(mask, a_2s)
    print("Minimum = (" + str(min_a_2) + ", " + str(min_var) + ")")

    plt.figure()
    plt.plot(a_2s, vars)
    plt.scatter(min_a_2, min_var, c="r", label="Minimum RMS Error")
    plt.xlabel(r"$a_2$")
    plt.ylabel(r"$\sqrt{\int_{-1}^1 | u_3(x; a_2) - \cos{(x)} |^2dx}$")
    plt.title(r"$u_3(x; a_2)$ RMS Error")
    plt.legend()
    if Save_figure:
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
    if Save_figure:
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
    plt.plot(x, trunc(-0.23293172690763075, x), label=r"$u_3(x;a_2=-0.233...)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Analytic and numerical solutions for $\frac{d^2y}{dx^2} + y = 0$")
    plt.legend()
    if Save_figure:
        plt.savefig("Approximations.png", dpi=1000)
    plt.show()


if plotD:
    x = np.linspace(-1, 1, 250)
    

    def Tau(x):
        return 1 - 2 * x**2 / 5
    
    Tau_y, cos_y = Tau(x), np.cos(x)

    # Mean error

    mean_err = quad(lambda x: (Tau(x) - np.cos(x)), -1, 1)[0]/2
    print(mean_err)

    # Plotting

    plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))

    plt.plot(x, Tau_y, label=r"$1-2x^2/5$", color ="r")
    plt.plot(x, cos_y, linestyle="--", label=r"$\cos{(x)}$", color="black")
    plt.fill_between(x, Tau_y, cos_y, alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Tau method solution for $\frac{d^2y}{dx^2} + y = 0$")
    plt.legend()

    plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))

    plt.plot(x, Tau_y - cos_y, color="r")
    plt.plot(x, np.zeros_like(x), linestyle="--", color="black")
    plt.fill_between(x, Tau_y-cos_y, np.zeros_like(x), alpha=0.5)
    plt.plot(x, mean_err*np.ones_like(x), label="Mean error")

    plt.xlabel("x")
    plt.ylabel("Error")

    plt.text(-1 + 0.01, mean_err + 0.002, "Mean error")

    if Save_figure:
        plt.savefig("u3.png", dpi=1000)
    plt.show()

    
