import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpmath import ff
import matplotlib as mpl

plt.rcParams["font.family"] = "Times New Roman"

fig, axs = plt.subplots(1, 1)

# Animated plot params

Period = 10
FPS = 60

frames = int(Period * FPS)
interval = 1000 / FPS  # frametime in ms

# Setting x, y limits
axs.set_xlim(-1, 1)
axs.set_ylim(-1, 1)
axs.set_aspect("equal")
axs.set_gid(True)

axs.set_xlabel("x")
axs.set_ylabel("y")


# Gridlines
axs.axhline(0, color="black", linewidth=1)
axs.axvline(0, color="black", linewidth=1)

axs.set_xticks(np.linspace(-1, 1, 5))
axs.set_yticks(np.linspace(-1, 1, 5))


# Colourbar

cmap = plt.get_cmap("brg", frames)

norm = mpl.colors.Normalize(vmin=0, vmax=2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(
    sm,
    ticks=np.linspace(0, 2, 5),
    ax=axs,
    label=r"$\alpha$",
)


# Plotting
t = np.linspace(-1, 1, 250)

axs.plot(t, t**3, label=r"$y=x^3$", color=cmap(0))
axs.plot(t, 3 * t**2, label=r"$y = 3 x^2$", color=cmap(int(frames/2)))
axs.plot(t, 6 * t, label=r"$y = 6 x$", color=cmap(frames))

axs.legend()


# Caputo

t_short = np.linspace(0, 1, 250)

def Poly_Caputo(t, n, alpha):
        # x^n
        if alpha > n:
            return np.zeros_like(t)
        else:
            return ff(n, alpha) * t ** (n - alpha)

# Animated plot


# initializing a line variable
(line,) = axs.plot([], [], lw=3)

# data which the line will
# contain (x, y)
def init():
    line.set_data([], [])
    return (line,)

def D_alpha_poly(t, n, alpha):
    # x^n
    if alpha > n:
        return np.zeros_like(t)
    else:
        return ff(n, alpha) * t ** (n - alpha)


# Integer order derivatives of x^3
axs.plot(t_short, t_short**3, linestyle="--", color=cmap(0))
axs.plot(
    t_short, 3 * t_short**2, linestyle="--", color=cmap(int(frames / 2))
)
axs.plot(t_short, 6 * t_short, linestyle="--", color=cmap(frames))

annot = axs.annotate(r'$\alpha$: 0', (0.5, -0.5))

def animate(i):
    I = 2 * i / frames

    y = D_alpha_poly(t_short, 3, I)

    line.set_data(t_short, y)
    line.set_color(cmap(i))
    annot.set_text(r'$\alpha$: ' + str(np.round(I, 3)))
    return (line,)

anim = FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=interval, blit=True
)

anim.save("Oscillators/polyOverlay.mp4", dpi=250)

