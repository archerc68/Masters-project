import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.special import gammaln, loggamma, gamma, poch, rgamma, gammaln
from pymittagleffler import mittag_leffler
from scipy.integrate import quad
from scipy.linalg import lu_factor, lu_solve
import mpmath as mp


plt.rcParams["font.family"] = "Times New Roman"


# ----------- Input parameters ---------- #

"Solves FDEs (Caputo) using the spectral method (Doha et al.) in the form:"


"D^alpha y(x) = sum_j [d_j D^beta_j y(x)] + d_(k+1) y(x) + d_(k+2) g(x)"

"alpha > beta_0 > beta_1 > ... > beta_k > 0"


"L:             x in [0, L]"
"m:             No. Chebyshev poly. used"
"alpha:         Leading fractional derivative order"
"beta_k         RHS fractional derivative orders"
"d_j:           RHS FD coefficients"
"a_i:           a_i = y^(a_order[i])(0) -- Boundary conditions"
"b_i:           b_i = y^(b_order[i])(L) -- Boundary conditions"
"g(x):          RHS perturbing function"


# ------------- Constants -------------- #

hbar = 6.62607015e-34 / (2 * np.pi)

# ------------- FDE Params ------------- #

L = 20
m = 20
beta = 1.85
beta_k = np.array([])


G = False


def g(x):
    return x * x


omega = 1
d_j = np.array([-(omega**2), 0])


# -------- Boundary conditions --------- #

# At x = 0
a_order = np.array([0, 1], dtype=int)
a_i = np.array([1, 0])


# At x = L
b_order = np.array([], dtype=int)
b_i = np.array([])


# region auxilliary parameters

n = int(np.ceil(beta))
k = len(beta_k)

# Debug
if len(beta_k) > 0:
    assert beta > np.max(beta_k)
assert len(d_j) == k + 2
assert len(a_i) + len(b_i) == n

# Auxilliary parameters
# x = np.linspace(0, L, 250)
# t = 2 * x / L - 1

N = 250
j = np.arange(N + 1)
t = -np.cos(np.pi * j / N)
x = (t + 1) * L / 2


# endregion

def KahanK(input, N):
    
    # Kahan summation algorithm
    sum = np.zeros((N+1, N+1))
    c = np.zeros((N+1, N+1))

    for i in range(input.shape[2]):
        y = input[:, :, i] - c
        t = sum + y
        c = (t - sum) - y
        sum = t.copy()
    return sum



# region D matrix
def D_1(N):
    D_matrix_T = np.zeros((N + 1, N + 1))
    k = np.arange(1, N + 1, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N + 1), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4 / L
    return np.array(D_matrix)


def D1(N, nu):
    if type(nu) is int:
        return np.linalg.matrix_power(D_1(N), nu)
    else:
        LB = int(np.ceil(nu))

        i = np.arange(N + 1, dtype=int)[:, None]
        j = np.arange(N + 1, dtype=int)[None, :]
        D_matrix = np.zeros((N + 1, N + 1))

        eps_j = np.ones((N + 1, N + 1))
        eps_j[:, 0] = 2

        coeff = 2 * i / (eps_j * L**nu)
        sign = np.where((i - LB) % 2 == 0, 1, -1)

        for k in range(LB, N + 1):
            a = k - nu + 1

            # Numerator & denominator
            log_num = loggamma(i + k) + loggamma(k - nu + 0.5)
            log_den = loggamma(k + 0.5) + gammaln(i - k + 1) + 2 * loggamma(a)

            num_den = np.exp(log_num - log_den)

            # Corrective terms to allow logarithms
            # [loggamma(k - j - nu + 1) woud return errors]
            # Terms derived from gamma(a + j) * gamma(a - j)

            factors = (a + j - 1) / (a - j)
            factors[:, 0] = 1
            term = np.cumprod(factors, axis=1)

            iteration = num_den * coeff * sign / term
            sign *= -1

            # Masking values
            iteration = np.where(k <= i, iteration, 0)
            D_matrix += iteration

        return D_matrix


# endregion

# region New D


def I_plus(i, n):
    def I_plusn(i):
        numI = -i * (i + n - 1)
        denI = (i - 1) * (i - n)
        return numI/denI
    return np.where(i > n, I_plusn(i), 1)


def J_plus(j, nu, n):
    numJ = (-j + n - nu + 1)
    denJ = (j + n - nu)

    ansJ = numJ/denJ
    ansJ[j == 0] = 1
    return ansJ


def K_plus(i, j, k, nu):
    def k_plusi(k):
        numK = (i * i - (k - 1) ** 2) * (2 * k - 2 * nu - 1)
        denK = (j * j - (k - nu) ** 2) * (2 * k - 1)
        return numK / denK

    ansK = np.where(k <= i, k_plusi(k), 0)
    ansK[:, :, 0] = 1

    return ansK

def seed(nu, n):
    num = 2 ** (2 * n - 1) * gamma(n + 1) * gamma(-nu + n + 0.5)
    den = np.sqrt(np.pi) * gamma(-nu + n + 1) ** 2
    return num / den


def D(N, nu):
    if type(nu) is int:
        return np.linalg.matrix_power(D_1(N), nu)
    else:
        n = int(np.ceil(nu))
        Mat = np.zeros((N + 1, N + 1))

        arrij = np.arange(N + 1)
        arrk = np.arange(N + 1 - n) + n

        I, J = I_plus(arrk, n), J_plus(arrij, nu, n)

        i = arrij[:, None, None]
        j = arrij[None, :, None]
        k = arrk[None, None, :]

        K = K_plus(i, j, k, nu)

        Mat[n:, 0] = seed(nu, n) * np.cumprod(I)

        Mat = Mat[:, 0][:, None] * np.cumprod(J)[None, :]

        Mat = Mat * KahanK(np.cumprod(K, axis=2), N)

        # Prefactors
        eps_j = np.ones((N+1, N+1))
        eps_j[:, 0] = 2
        coeff = 2 / (L ** nu * eps_j)

        return coeff * Mat


# endregion


# region Solving FDE


# Solving
def Solve_Tau(m, D):
    # Phi(x)
    phi = chebvander(t, m).T
    phi_0 = phi[:, 0]
    phi_L = phi[:, -1]

    phi_BC = np.empty((n, m + 1))
    phi_BC[: len(a_i), :] = phi_0
    phi_BC[len(a_i) :, :] = phi_L

    # G_T

    if G:
        G_T = least_squares(lambda G_T: G_T @ phi - g(x), np.random.random(m + 1)).x

        # Plotting G^T
        plt.figure(1).add_axes((0.1, 0.3, 0.8, 0.6))
        gvals = g(x)
        plt.plot(x, gvals, label="g(x)")
        approx = G_T @ phi
        plt.plot(x, approx, linestyle="--", label="G^T phi(x)")
        plt.ylabel("y")
        plt.title("Fitted G^T (m = " + str(m) + ")")
        plt.legend()
        plt.figure(1).add_axes((0.1, 0.1, 0.8, 0.2))
        plt.xlabel("x")
        plt.ylabel("deviation")
        plt.plot(x, approx - gvals)
        plt.plot(x, np.zeros_like(x), linestyle="--")
        # plt.savefig("close.png")
        plt.show()

    else:
        G_T = np.zeros(m + 1)

    # D'
    D_alpha = D(N=m, nu=beta)
    D_beta_sum = np.zeros((m + 1, m + 1))
    for i in range(k):
        D_beta_sum += d_j[i] * D(N=m, nu=beta_k[i])
    D_prime = D_alpha - D_beta_sum - d_j[k] * np.eye(m + 1)

    # Operating matrix
    Operator = np.empty((m + 1, m + 1))
    Operator[:, :] = D_prime[:, :]

    # Boundary conditions
    D_order = np.concatenate((a_order, b_order), dtype=int)

    for i in range(n):
        order = int(D_order[i])
        Operator[:, m - n + 1 + i] = D(N=m, nu=order) @ phi_BC[i]

    # Column vector
    column_vec = np.empty(m + 1)
    column_vec[: m - n + 1] = G_T[: m - n + 1]
    column_vec[m - n + 1 :] = np.concatenate((a_i, b_i))

    lu, piv = lu_factor(Operator.T)
    C = lu_solve((lu, piv), column_vec.T)
    y = C.T @ phi

    return y, C


y, C = Solve_Tau(m, D)
y1, _ = Solve_Tau(m, D1)

# endregion

# region Analytical solution


def analytic(t, beta, omega, y_0, y_dot_0):
    omega_t_pow = -(omega**2) * t**beta

    f1 = mittag_leffler(omega_t_pow, beta, 1)
    f2 = t * mittag_leffler(omega_t_pow, beta, 2)

    return np.real(y_0 * f1 + y_dot_0 * f2)


# endregion

# region Error analysis


def err(m, D):
    _, C = Solve_Tau(m, D)

    def diff_square(x):
        a_vals = analytic(x, beta, omega, a_i[0], a_i[1])
        t = 2 * x / L - 1
        phi = chebvander(t, m).T
        return (a_vals - C.T @ phi) ** 2

    RMSE = np.sqrt(quad(diff_square, 0, L)[0])
    return RMSE

ms = np.arange(2, 50)
RMSEs = np.vectorize(err)(ms, D)
RMSEs1 = np.vectorize(err)(ms, D1)

plt.figure()
plt.semilogy()
plt.scatter(ms, RMSEs, alpha=0.5, color="red", label="New")
plt.scatter(ms, RMSEs1, alpha=0.5, color="blue", label="Old")

plt.xlabel(r"N")
plt.ylabel("RMSE")
plt.legend()
plt.title(r"Error in $u_N$ approximation for fractional pendulum")

# plt.savefig("Error_fractional.png", dpi=1000)
plt.show()


# endregion


# ---------- Plotting output ---------- #

if __name__ == "__main__":
    analytic_vals = analytic(x, beta, omega, a_i[0], a_i[1])
    plt.figure(2).add_axes((0.1, 0.3, 0.8, 0.6))
    y, _ = Solve_Tau(22, D)
    plt.plot(x, y, label="Tau (spectral) method")
    plt.plot(x, analytic_vals, linestyle="--", label="Analytical solution")
    plt.legend()
    plt.ylabel("y")

    plt.figure(2).add_axes((0.1, 0.1, 0.8, 0.2))
    plt.xlabel("t")
    plt.ylabel("deviation")
    plt.plot(x, y - analytic_vals)
    plt.plot(x, np.zeros_like(x), linestyle="--")
    # plt.savefig("y.png")
    plt.show()
