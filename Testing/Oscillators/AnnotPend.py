import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

omega_0 = 4
gamma = 0.25
l = 1
theta_0 = np.arcsin(0.55/l)


Period = 10
FPS = 300
frames = FPS*Period

def Pend(t):

    omega = np.sqrt(omega_0 * omega_0 + gamma * gamma)
    oscillation = theta_0 * np.cos(omega * t)

    decay = np.exp(-gamma * t)

    return decay * oscillation


ts = np.linspace(0, Period, frames)
thetas = Pend(ts)
x, y = l * np.sin(thetas), - l * np.cos(thetas)


# Animation

# fig = plt.figure()
# axis = plt.axes(xlim=(-1.1*l, 1.1*l), ylim=(-1.1*l, 0), aspect="equal")

fig, axis = plt.subplots(1, 1)

axis.set_aspect("equal")
axis.set_xlim(-0.6, 0.6)
axis.set_ylim(-1.2, 0)
axis.axis("off")


axis.plot(np.zeros(2), np.array([0, -0.25*l]), linestyle="--", color="black")


(line,) = axis.plot([], [], lw=3)
(bob,) = axis.plot([], [], "ro", ms=6)

def init():
        line.set_data([], [])
        bob.set_data([], [])
        return line, bob

def animate(i):
      line.set_data(np.array([0, x[i]]), np.array([0, y[i]]))
      bob.set_data(np.array([x[i]]), np.array([y[i]]))
      return line, bob

anim = FuncAnimation(
    fig, animate, init_func=init, frames=frames, interval=1000/FPS, blit=True
)

# anim.save("Oscillators/AnnotPend.mp4", dpi=250)

plt.show()