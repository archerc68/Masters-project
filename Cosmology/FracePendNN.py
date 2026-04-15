import torch
from FDEint import FDEint
import matplotlib.pyplot as plt
from pymittagleffler import mittag_leffler
import numpy as np
from sklearn.metrics import mean_squared_error
from fodeint import caputoEuler

omega_0, q_0, p_0 = 2, 1, 0
m = 1


def Fdamp(t, alpha):

    omega_t_pow = -(omega_0**2) * t ** (2 * alpha)

    f1 = mittag_leffler(omega_t_pow, 2 * alpha, 1)
    f2 = t**alpha * mittag_leffler(omega_t_pow, 2 * alpha, alpha + 1)

    if alpha > 0.5:
        q = np.real(q_0 * f1 + p_0 / m * f2)
    else:
        q = np.real(q_0 * f1)
    return q


def fractional_diff_eq(t, x):
    return -(omega_0**2) * x


t = torch.linspace(0.0, 10.0, 2001).unsqueeze(-1).unsqueeze(0)
y0 = torch.tensor([q_0, q_0]).unsqueeze(0)
alpha = torch.tensor([0.975])

solution = FDEint(fractional_diff_eq, t, y0, alpha)

t_numpy = np.linspace(0, 10, 2001)

y_pred = solution.squeeze().numpy()[:, 0]
y_true = Fdamp(t_numpy, alpha.numpy()/2)


def f(y, t):
    return -omega_0**2*y

y = caputoEuler(alpha.numpy(), f, q_0, t_numpy)


plt.figure()
plt.plot(t_numpy, y_pred)
plt.plot(t_numpy, y_true)
plt.plot(t_numpy, y)
plt.show()

print("RMSE = " + str(np.sqrt(mean_squared_error(y_true, y_pred))))
print("RMSE L1 = " + str(np.sqrt(mean_squared_error(y_true, y))))
