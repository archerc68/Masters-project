import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymittagleffler import mittag_leffler

t_arr = np.linspace(0, 10, 250)

# Fractional pendulum in all regimes
def Fpend(t, alpha):
    # alpha > 0.5

    omega_0 = 1
    y_0 = 1
    y_dot_0 = 0

    # bc = [y_0, y_dot_0]

    def ml(z, alpha, beta):
        return mittag_leffler(z, alpha, beta)

    ml_vec = np.vectorize(ml)

    omega_t = -((omega_0 * t**alpha) ** 2)

    y = y_0 * ml_vec(omega_t, 2 * alpha, 1)
    y += y_dot_0 * t * ml_vec(omega_t, 2 * alpha, 2)

    return np.real(y)

# Underdamped classical pendulum
def Cpend(t, gamma):
    omega_0 = 1
    y_0 = 1
    y_dot_0 = 0

    omega = np.sqrt(omega_0**2 - gamma**2)

    decay = np.exp(-gamma * t)
    oscillation = y_0 * np.cos(omega * t) + (2 * gamma + 1) * y_dot_0 * np.cos(
        omega * t
    )

    return decay * oscillation

# Overdamped classical pendulum
def CpendOver(t, lgamma):

    gamma = 10**lgamma
    omega_0 = 1
    y_0 = 1
    y_dot_0 = 0

    omega = np.sqrt(gamma**2 - omega_0**2)

    C = y_dot_0/(omega-gamma)

    A = (y_0 + C)/2
    B = (y_0 - C)/2

    return A * np.exp((omega-gamma)*t) + B*np.exp((-omega-gamma)*t)


# Plotting

# back plot
def OscPlot(pend, alpha_arr, axs, cmap, norm):
    
    t, alpha = np.meshgrid(t_arr, alpha_arr)

    y1 = pend(t, alpha)

    axs.plot_surface(
        t, alpha, y1, cmap=cmap, norm=norm, edgecolor="none", linewidth=0, shade=True,
    )

# Intersecring plane
def plane(alphaInt, axs, pcolor):
    t_plane, y_plane = np.linspace(0, 10, 2), np.linspace(-1, 1, 2)
    T_plane, Y_plane = np.meshgrid(t_plane, y_plane)
    alpha_plane = np.full_like(T_plane, alphaInt)


    axs.plot_wireframe(
        T_plane,
        alpha_plane,
        Y_plane,
        color=pcolour,
        alpha=0.8,
    )

    axs.plot(t_arr, alphaInt, pend(t_arr, alphaInt), color=pcolour)

# Plane intersection plot
def IntersectPlot(alphaInt, alphaMin, alphaMax, pend, axs, cmap, norm, pcolour):
    alpha_arr1 = np.linspace(
    alphaInt, alphaMax, int(alpha_points * (1 - alphaInt) / (alphaMax - alphaMin))
    )

    OscPlot(pend, alpha_arr1, axs, cmap, norm) # Back Plot


    plane(alphaInt, axs, pcolour)  # Plane


    alpha_arr2 = np.linspace(
        alphaMin, alphaInt, int(alpha_points * alphaInt / (alphaMax - alphaMin))
    )

    OscPlot(pend, alpha_arr2, axs, cmap, norm) # Front plot

# Grid Plot


norm = mcolors.Normalize(-1, 1)

cmapC = plt.get_cmap("viridis")
cmapF = plt.get_cmap("inferno")

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})

axs[0, 1].invert_yaxis()
axs[1, 1].invert_yaxis()

OscPlot(Cpend, np.linspace(0, 1, 250), axs[0, 0], cmapC, norm)
OscPlot(CpendOver, np.linspace(0, 2, 250), axs[1, 0], cmapC, norm)

OscPlot(Fpend, np.linspace(0.5, 1, 250), axs[0, 1], cmapF, norm)
OscPlot(Fpend, np.linspace(0, 0.5, 250), axs[1, 1], cmapF, norm)

# Axis labels
for i in (range(2)):
    axs[i, 1].set_ylabel(r"$\alpha$")
    for j in (range(2)):
        axs[i, j].set_xlabel("t")
        axs[i, j].set_zlabel("y")
        # axs[i, j].grid(False)

axs[0, 0].set_ylabel(r"$\zeta$")
axs[1, 0].set_ylabel(r"$\log_{10}{\zeta}$")

# Ticks
for j in (range(2)):
    axs[0, j].set_zticks([-1, 0, 1])
    axs[1, j].set_zticks([0.0, 0.5, 1.0])

axs[0, 1].set_yticks([0.50, 0.75, 1.0])
axs[1, 1].set_yticks([0.0, 0.25, 0.50])

# Titles
axs[0, 0].set_title("Classical")
axs[0, 1].set_title("Fractional")


# plt.savefig("Figures/QuadSHM.svg")

plt.show()
