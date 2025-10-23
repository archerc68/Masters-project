import numpy as np
import matplotlib.pyplot as plt


# Going up to alpha (h=1)
def GLkernal(alpha):
    n = int(np.ceil(alpha))
    k = np.arange(n + 1)
    g_k = np.empty(n + 1)
    g_k[1:] = (alpha + 1) / k[1:] - 1
    return 2**alpha - np.sum(np.cumprod(g_k))

print(GLkernal(0.2))

alphas = np.linspace(0, 1, 50)
y = [GLkernal(i) for i in alphas]

plt.figure()
plt.plot(alphas, y)
plt.show()
