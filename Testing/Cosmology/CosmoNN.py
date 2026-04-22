import torch
from FDEint import FDEint
import matplotlib.pyplot as plt
import numpy as np



def fractional_diff_eq(t, a):
    return a*np.sqrt(18*a**(-3) + 0.2*a**(-4))

def classical_diff_eq(t, a):
    return a*np.sqrt(0.2*a**(-3) + 1e-4*a**(-4) + 0.8)

eps = 1e-4

t = torch.linspace(eps, 1.0, 2001).unsqueeze(-1).unsqueeze(0)
a0 = torch.tensor([eps, eps]).unsqueeze(0)
alpha = torch.tensor([0.5])

FF = FDEint(fractional_diff_eq, t, a0, alpha)
CF = FDEint(classical_diff_eq, t, a0, torch.tensor([1]))

fig, axs = plt.subplots(1)
axs.loglog(t.squeeze(), FF.squeeze())
axs.loglog(t.squeeze(), CF.squeeze())
plt.show()

