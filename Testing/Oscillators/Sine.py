import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpmath import ff
import matplotlib as mpl
from pymittagleffler import mittag_leffler


plt.rcParams["font.family"] = "Times New Roman"

fig, axs = plt.subplots(1, 1)

# Animated plot params

Period = 10
FPS = 60

frames = int(Period * FPS)
interval = 1000 / FPS  # frametime in ms

# Setting x, y limits
axs.set_xlim(-2 * np.pi, 2 * np.pi)
axs.set_ylim(-2, 2)
# axs.set_aspect("equal")
axs.set_gid(True)

axs.set_xlabel("x")
axs.set_ylabel("y")


# Gridlines
axs.axhline(0, color="black", linewidth=1)
axs.axvline(0, color="black", linewidth=1)

axs.set_xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
axs.set_yticks(np.linspace(-2, 2, 5))


# Colourbar

cmap = plt.get_cmap("brg", frames)

norm = mpl.colors.Normalize(vmin=0, vmax=2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(
    sm,
    ticks=np.linspace(0, 2, 5),
    ax=axs,
    label=r"$\alpha$",
    orientation="horizontal",
)


# Plotting
t = np.linspace(-2*np.pi, 2*np.pi, 250)

axs.plot(t, np.sin(t), label=r"$y=\sin{(x)}$", color=cmap(0))
axs.plot(t, np.cos(t), label=r"$y = \cos{(x)}$", color=cmap(int(frames/2)))
axs.plot(t, -np.sin(t), label=r"$y = -\sin{(x)}$", color=cmap(frames))

axs.legend(loc="lower left")


# Caputo

t_short = np.linspace(0.1, 2*np.pi, 250)

# Animated plot


# initializing a line variable
(line,) = axs.plot([], [], lw=3)

# data which the line will
# contain (x, y)
def init():
    line.set_data([], [])
    return (line,)

def D_alpha_sin(t, alpha, omega=1):
    omega_t = omega*t*1j
    den = 2j*t**alpha
    num = mittag_leffler(omega_t, 1, 1-alpha) - mittag_leffler(-omega_t, 1, 1-alpha)
    return np.real(num/den)


annot = axs.annotate(r'$\alpha$: 0', (4, -1.5))

def animate(i):
    I = 2 * i / frames

    y = D_alpha_sin(t_short, I)

    line.set_data(t_short, y)
    line.set_color(cmap(i))
    annot.set_text(r'$\alpha$: ' + str(np.round(I, 3)))
    return (line,)

anim = FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=interval, blit=True
)

anim.save("Oscillators/sinOverlay.mp4", dpi=250)
