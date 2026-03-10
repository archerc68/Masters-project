import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda")

gamma = 0.5
omega = 1


def pde(x, y):
    d2y_dx2 = dde.grad.hessian(y, x)
    dy_dx = dde.grad.jacobian(y, x)
    return d2y_dx2 + 2 * gamma * dy_dx + omega * omega * y


def exact_sol(x):
    decay = np.exp(-gamma * x)
    omega_pert = np.sqrt(omega**2 + gamma**2)
    osc = np.cos(omega_pert * x) + gamma / omega * np.sin(omega_pert * x)
    return decay * osc


geom = dde.geometry.Interval(0, 6 * np.pi)


def on_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


# Dirichlet: f(0) = 1
bc_dirichlet = dde.icbc.DirichletBC(geom, lambda x: 1.0, on_left)

# Neumann: f'(0) = 0  (normal derivative = 0 at x=0)
bc_neumann = dde.icbc.NeumannBC(geom, lambda x: x, on_left)

data = dde.data.PDE(
    geom, pde, [bc_neumann, bc_dirichlet], 128, 2, solution=exact_sol, num_test=100
)

# NN
layer_size = [1] + [75] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"

net = dde.nn.FNN(layer_size, activation, initializer).to(device)

# Compiling
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

loss_history, loss_state = model.train(iterations=10000)
dde.saveplot(loss_history, loss_state, isplot=True)

# Plotting
x = geom.uniform_points(250, True)
y_pred = model.predict(x)
y_exact = exact_sol(x)

plt.figure()
plt.plot(x, y_exact)
plt.plot(x, y_pred)
plt.show()
