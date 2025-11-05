import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.special import gamma
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Nodes consistent with affine map
# ---------------------------------------------------------------
def cheb_nodes_lobatto(N, L=1.0):
    """
    Chebyshev–Gauss–Lobatto nodes mapped to [0,L].
    Returns (x, t): physical and computational nodes.
    """
    j = np.arange(N+1)
    t = np.cos(np.pi * j / N)        # nodes in [-1,1]
    x = 0.5 * L * (t + 1.0)          # map to [0,L]
    return x, t

# ---------------------------------------------------------------
# Fractional operator placeholder
# ---------------------------------------------------------------
def fractional_op_matrix(N, alpha):
    """
    Placeholder operational matrix for D_t^alpha in Chebyshev basis.
    Replace with explicit formulas from the literature for accuracy.
    """
    D = np.zeros((N+1, N+1))
    for n in range(N+1):
        if n > 0:
            D[n,n] = gamma(n+1)/gamma(n+1-alpha)
    return D

# ---------------------------------------------------------------
# Multi-term FDE solver
# ---------------------------------------------------------------
def solve_fde(N, L, alphas, coeffs, f_fun, bc=(0,0)):
    # nodes
    x, t = cheb_nodes_lobatto(N, L)

    # Vandermonde in computational variable t
    V = chebvander(t, N)

    # combined operator in coefficient space
    Dsum = np.zeros((N+1, N+1))
    for c,a in zip(coeffs, alphas):
        Dsum += c * (2.0/L)**a * fractional_op_matrix(N, a)

    # collocation system
    fvals = f_fun(x)
    A = V @ Dsum
    b = fvals

    # enforce Dirichlet BCs at x=0,L (t=-1,1)
    Vleft = chebvander(np.array([-1.0]), N)
    Vright = chebvander(np.array([1.0]), N)
    A = np.vstack([A, Vleft, Vright])
    b = np.concatenate([b, np.array(bc)])

    # solve for coefficients
    a = np.linalg.lstsq(A, b, rcond=None)[0]
    u_vals = V @ a
    return x, u_vals, a

# ---------------------------------------------------------------
# Example
# ---------------------------------------------------------------
if __name__ == "__main__":
    L = 1.0
    N = 64
    alphas = [0.9, 0.5]
    coeffs = [1.0, -0.7]
    f_fun = lambda x: np.exp(x)

    x, u, a = solve_fde(N, L, alphas, coeffs, f_fun, bc=(0,0))
    print("x:", x)
    print("u(x):", u)

    plt.figure()
    plt.plot(x, u)
    plt.show()