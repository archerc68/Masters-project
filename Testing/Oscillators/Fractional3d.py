import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymittagleffler import mittag_leffler

t_points = int(5e1)
alpha_points = int(5e1)

alphaInt = 0.75  # Plane alpha value

alphaMin, alphaMax = 0.5, 1


def pend(t, alpha):
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


# Plotting

fig = plt.figure(figsize=plt.figaspect(0.5))

gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
ax3d = fig.add_subplot(gs[0], projection="3d", aspect="auto")
ax2d = fig.add_subplot(gs[1], aspect=5 * 3 / 4)

ax3d.grid(False)
ax3d.invert_yaxis()
# ax3d.set_ylim(1, 0)
# axs.set_axis_off()

norm = mcolors.Normalize(-1, 1)
cmap = plt.get_cmap("inferno")
pcolour = "indigo"

# back plot
t_arr = np.linspace(0, 10, t_points)
alpha_arr1 = np.linspace(
    alphaInt, alphaMax, int(alpha_points * (1 - alphaInt) / (alphaMax - alphaMin))
)

t1, alpha1 = np.meshgrid(t_arr, alpha_arr1)

y1 = pend(t1, alpha1)

surf1 = ax3d.plot_surface(
    t1, alpha1, y1, cmap=cmap, norm=norm, edgecolor="none", linewidth=0
)

# Intersecring plane

t_plane, y_plane = np.linspace(0, 10, 2), np.linspace(-1, 1, 2)
T_plane, Y_plane = np.meshgrid(t_plane, y_plane)
alpha_plane = np.full_like(T_plane, alphaInt)


plane = ax3d.plot_wireframe(
    T_plane,
    alpha_plane,
    Y_plane,
    color=pcolour,
    alpha=0.8,
)

ax3d.plot(t_arr, alphaInt, pend(t_arr, alphaInt), color=pcolour)

# Front plot
alpha_arr2 = np.linspace(
    alphaMin, alphaInt, int(alpha_points * alphaInt / (alphaMax - alphaMin))
)

t2, alpha2 = np.meshgrid(t_arr, alpha_arr2)

y2 = pend(t2, alpha2)

surf2 = ax3d.plot_surface(
    t2, alpha2, y2, cmap=cmap, norm=norm, edgecolor="none", linewidth=0
)

ax3d.set_xlabel("t")
ax3d.set_ylabel(r"$\alpha$")
ax3d.set_zlabel("y")

ax3d.set_zlim(-1, 1)

# 2d plot
t_2d = np.linspace(t_arr[0], t_arr[-1], 250)
ax2d.set_ylim(-1.1, 1.1)

ax2d.axhline(0, 0, 10, linestyle="--", color="k")
ax2d.plot(t_2d, pend(t_2d, alphaInt), color=pcolour)

ax2d.set_xlabel("t")
# ax2d.set_ylabel("y(t)")

ax2d.set_title(r"$\alpha=$" + str(alphaInt))

plt.savefig("Figures/FractionalPend.svg")

plt.show()
