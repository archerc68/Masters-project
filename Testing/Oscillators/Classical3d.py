import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def pend(t, gamma):
    omega_0 = 1
    y_0 = 1
    y_dot_0 = 0

    omega = np.sqrt(omega_0**2 - gamma**2)

    decay = np.exp(-gamma * t)
    oscillation = y_0 * np.cos(omega * t) + (2 * gamma + 1) * y_dot_0 * np.cos(
        omega * t
    )

    return decay * oscillation

t_points = int(5e2)
gamma_points = int(5e2)

# Plotting

fig = plt.figure(figsize=plt.figaspect(.5))

gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
ax3d = fig.add_subplot(gs[0], projection="3d", aspect="auto")
ax2d = fig.add_subplot(gs[1], aspect=5*3/4)

ax3d.grid(False)
# axs.set_axis_off()

norm = mcolors.Normalize(-1, 1)
cmap = plt.get_cmap("viridis")

gint = 0.5  # Plane gamma value

# back plot
t_arr = np.linspace(0, 10, t_points)
gamma_arr1 = np.linspace(0, gint, int(gamma_points*gint/1))

t1, gamma1 = np.meshgrid(t_arr, gamma_arr1)

y1 = pend(t1, gamma1)

surf1 = ax3d.plot_surface(
    t1, gamma1, y1, cmap=cmap, norm=norm, edgecolor="none", linewidth=0
)

# Intersecring plane

t_plane, y_plane = np.linspace(0, 10, 2), np.linspace(-1, 1, 2)
T_plane, Y_plane = np.meshgrid(t_plane, y_plane)
gamma_plane = np.full_like(T_plane, gint)

pcolour = "red"
plane = ax3d.plot_wireframe(
    T_plane,
    gamma_plane,
    Y_plane,
    color=pcolour,
    alpha=0.8,
)

ax3d.plot(t_arr, gint, pend(t_arr, gint), color=pcolour)

# Front plot
gamma_arr2 = np.linspace(gint, 1, int(gamma_points*(1-gint)))

t2, gamma2 = np.meshgrid(t_arr, gamma_arr2)

y2 = pend(t2, gamma2)

surf2 = ax3d.plot_surface(
    t2, gamma2, y2, cmap=cmap, norm=norm, edgecolor="none", linewidth=0
)

ax3d.set_xlabel("t")
ax3d.set_ylabel(r"$\zeta$")
ax3d.set_zlabel("y")

ax3d.set_zlim(-1, 1)

# 2d plot
t_2d = np.linspace(t_arr[0], t_arr[-1], 250)

ax2d.plot(t_2d, pend(t_2d, gint), color=pcolour)

ub = np.exp(-gint*t_2d)
lb = -np.exp(-gint*t_2d)

ax2d.plot(t_2d, ub, linestyle="--", color="k")
ax2d.plot(t_2d, lb, linestyle="--", color="k")

ax2d.fill_between(t_2d, lb, ub, alpha=0.8)

ax2d.set_xlabel("t")
# ax2d.set_ylabel("y(t)")

ax2d.set_title(r"$\zeta=$" + str(gint))

plt.savefig("Figures/ClassicalPend.svg")

plt.show()
