import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpmath import ff
from pymittagleffler import mittag_leffler

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

cos = False
damp = False
dampPhase = False
poly = True
cdamp = False

# MP4 params
Period = 10  # Length of clip in seconds
FPS = 60
DPI = 400

frames = int(Period * FPS)
interval = 1000 / FPS  # frametime in ms


# initializing a figure in
# which the graph will be plotted
fig = plt.figure()


if cos:
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0.05, 4 * np.pi), ylim=(-2, 2))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Fractional derivatives of $\sin{(x)}$")

    # initializing a line variable
    (line,) = axis.plot([], [], lw=3)

    # data which the line will
    # contain (x, y)
    def init():
        line.set_data([], [])
        return (line,)

    def D_alpha_sin(t, omega, alpha):
        f1 = mittag_leffler(1j * omega * t, 1, 1 - alpha)
        f2 = mittag_leffler(-1j * omega * t, 1, 1 - alpha)

        return np.real((f1 - f2) / (2j * t**alpha))
    
    annot = axis.annotate(r"$\alpha$: 0", (15, 1.5))

    def animate(i):
        t = np.linspace(0.05, 4 * np.pi, 1000)

        # plots a sine graph
        alpha = 2 * i / frames
        y = D_alpha_sin(t, 1, alpha)
        line.set_data(t, y)

        annot.set_text(r"$\alpha$: " + str(np.round(alpha)))

        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    anim.save(
        "Oscillators/cosFD1.mp4",
        writer="ffmpeg",
        fps=FPS,
        dpi=DPI,
    )


if damp:
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, 20), ylim=(-1, 1))

    plt.xlabel(r"$t$")
    plt.ylabel(r"$y(t)$")
    plt.title("Motion of fractional pendulum")

    cmap = plt.get_cmap("plasma_r", frames)

    norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(
        sm,
        ticks=np.linspace(0.5, 1, 5),
        ax=axis,
        label=r"$\alpha$",
    )

    # initializing a line variable
    (line,) = axis.plot([], [], lw=3)

    # data which the line will
    # contain (x, y)
    def init():
        line.set_data([], [])
        return (line,)

    def damped(t, omega, alpha, q_0, p_0, m):
        omega_t_pow = -(omega**2) * t ** (2 * alpha)

        f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
        f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

        if alpha > 0.5:
            q = np.real(q_0 * f1 + p_0 / m * f2)
            p = np.real(p_0 * f1 - m * omega**2 * q_0 * f2)
        else:
            q = np.real(q_0 * f1)
            p = np.real(-m * omega**2 * q_0 * f2)

        return q, p
    

    annot = axis.annotate(r'$\alpha$: 0.5', (15, 0.5))

    def animate(i):
        t = np.linspace(0, 20 * np.pi, 2000)

        # plots a sine graph
        alpha = 1 - 0.5 * i / frames
        q, _ = damped(t, 1, alpha, 1, 0, 1)
        line.set_data(t, q)
        line.set_color(cmap(frames - i))

        annot.set_text(r"$\alpha$: " + str(np.round(alpha, 2)))

        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    anim.save(
        "Oscillators/Damped.mp4",
        writer="ffmpeg",
        fps=FPS,
        dpi=DPI,
    )


if dampPhase:
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    axis.set_aspect("equal")

    plt.xlabel(r"$p_{\alpha}$")
    plt.ylabel(r"$q_{\alpha}$")
    plt.title("Phase space of fractional pendulum")

    cmap = plt.get_cmap("plasma_r", frames)

    norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(
        sm,
        ticks=np.linspace(0.5, 1, 5),
        ax=axis,
        label=r"$\alpha$",
    )

    # initializing a line variable
    (line,) = axis.plot([], [], lw=3)

    # data which the line will
    # contain (x, y)
    def init():
        line.set_data([], [])
        return (line,)

    def damped(t, omega, alpha, q_0, p_0, m):
        omega_t_pow = -(omega**2) * t ** (2 * alpha)

        f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
        f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

        if alpha > 0.5:
            q = np.real(q_0 * f1 + p_0 / m * f2)
            p = np.real(p_0 * f1 - m * omega**2 * q_0 * f2)
        else:
            q = np.real(q_0 * f1)
            p = np.real(-m * omega**2 * q_0 * f2)

        return q, p
    
    annot = axis.annotate(r'$\alpha$: 0.5', (0.75, 0.9))

    def animate(i):
        t = np.linspace(0, 20 * np.pi, 2000)

        # plots a sine graph
        alpha = 1 - 0.5 * i / frames
        q, p = damped(t, 1, alpha, 1, 0, 1)
        line.set_data(q, p)
        line.set_color(cmap(frames - i))

        annot.set_text(r"$\alpha$: " + str(np.round(alpha, 3)))

        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    anim.save(
        "Oscillators/DampedPhase.mp4",
        writer="ffmpeg",
        fps=FPS,
        dpi=DPI,
    )


if poly:
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, 2), ylim=(0, 8))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"$D^{\alpha}x^3$")

    cmap = plt.get_cmap("plasma_r", frames)

    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(
        sm,
        ticks=np.linspace(0, 2, 5),
        ax=axis,
        label=r"$\alpha$",
    )

    # initializing a line variable
    (line,) = axis.plot([], [], lw=3)

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

    t = np.linspace(0, 2, 250)

    # Integer order derivatives of x^3
    plt.plot(t, t**3, linestyle="--", color=cmap(0), label=r"$\alpha=0$")
    plt.plot(
        t, 3 * t**2, linestyle="--", color=cmap(int(frames / 2)), label=r"$\alpha=1$"
    )
    plt.plot(t, 6 * t, linestyle="--", color=cmap(frames), label=r"$\alpha=2$")

    annot = axis.annotate(r'$\alpha$: 0', (1.5, 0.75))

    def animate(i):
        I = 2 * i / frames

        y = D_alpha_poly(t, 3, I)

        line.set_data(t, y)
        line.set_color(cmap(i))
        annot.set_text(r'$\alpha$: ' + str(np.round(I, 3)))
        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    plt.legend()

    anim.save(
        "Oscillators/polyFD1.mp4",
        writer="ffmpeg",
        fps=FPS,
        dpi=DPI,
    )


if cdamp:
    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, 20), ylim=(-1, 1))

    plt.xlabel(r"$t$")
    plt.ylabel(r"$y(t)$")
    plt.title("Motion of classical pendulum")

    cmap = plt.get_cmap("plasma", frames)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(
        sm,
        ticks=np.linspace(0, 1, 5),
        ax=axis,
        label=r"$\gamma$",
    )

    # initializing a line variable
    (line,) = axis.plot([], [], lw=3)

    # data which the line will
    # contain (x, y)
    def init():
        line.set_data([], [])
        return (line,)

    def damped(t, omega_0, gamma, q_0, p_0, m):

        omega = np.sqrt(omega_0**2 + gamma**2)

        decay = np.exp(-gamma*t)

        p = np.zeros_like(t)

        f1 = q_0*np.cos(omega*t)
        f2 = (p_0/m + gamma*q_0)/omega * np.sin(omega*t)

        q = decay*(f1 + f2)

        return q, p
    

    annot = axis.annotate(r'$\gamma$: 0.000', (15, 0.5))

    def animate(i):
        t = np.linspace(0, 20 * np.pi, 2000)

        # plots a sine graph
        gamma = i / frames
        q, _ = damped(t, 1, gamma, 1, 0, 1)
        line.set_data(t, q)
        line.set_color(cmap(i))

        annot.set_text(r"$\gamma$: " + str(np.round(gamma, 3)))

        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=frames, interval=interval, blit=True
    )

    anim.save(
        "Oscillators/Classical_Damped.mp4",
        writer="ffmpeg",
        fps=FPS,
        dpi=DPI,
    )