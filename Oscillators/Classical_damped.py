import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.integrate import quad
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"

n = 2
x_j = np.array([0, 0])
alpha_j = np.array([0, 1])
d_j = np.array([1, 0])
L = 10


assert len(alpha_j) == n & len(alpha_j) == len(d_j)

# ODE
gamma = 0.5
omega_0 = 4


# [0, L]
def diff_mat(alpha, N):
    D_matrix_T = np.zeros((N, N))
    k = np.arange(1, N, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4 / L
    return np.linalg.matrix_power(D_matrix, alpha)


def solve(N, x_y):
    Big_mat = diff_mat(2, N) + 2 * gamma * diff_mat(1, N) + omega_0**2 * np.eye(N)

    Big_mat[:, -1] = (diff_mat(alpha_j[0], N) @ chebvander(-1, N - 1).T)[:, 0]
    Big_mat[:, -2] = (diff_mat(alpha_j[1], N) @ chebvander(-1, N - 1).T)[:, 0]

    Big_mat_inv = np.linalg.inv(Big_mat)

    column_vec = np.zeros(N)
    column_vec[-1], column_vec[-2] = d_j[0], d_j[1]

    a_i = column_vec @ Big_mat_inv

    x = np.linspace(0, L, 250)

    def approx(x):
        ts = 2 * x / L - 1
        return a_i @ chebvander(ts, N - 1).T

    def analytic(x):
        decay = np.exp(-gamma * x)
        omega = np.sqrt(omega_0**2 - gamma**2)
        sfactor = (d_j[0] * gamma + d_j[1]) / omega
        oscillation = np.cos(omega * x) + sfactor * np.sin(omega * x)
        return decay * oscillation

    def approx_err(x):
        return (approx(x) - analytic(x)) ** 2

    RMSE = np.sqrt(quad(approx_err, 0, L)[0])

    if x_y:
        return x, approx(x), RMSE
    else:
        return RMSE


def sf(x, p=2):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


Damped_plot = 0
Error_N = 1

Save_fig = 0

if Damped_plot:
    N = 100
    x, y, RMSE = solve(N, x_y=True)
    print(RMSE)
    exp_decay = np.exp(-gamma * x)

    # Plotting
    plt.figure()

    plt.plot(x, exp_decay, linestyle="--", color="black", label=r"$y = e^{-\gamma x}$")
    plt.plot(x, -exp_decay, linestyle="--", color="black")
    plt.fill_between(x, -exp_decay, exp_decay, alpha=0.5, color="blue")
    plt.plot(x, y, color="red", label="Damped oscillator")

    # plt.title(r"$y'' + $" + str(2 * gamma) + r"$y' + $" + str(omega_0**2) + r"$y = 0$")
    plt.title(r"$\gamma = $" + str(gamma) + ", " + r"$\omega_0 = $" + str(omega_0))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")

    if Save_fig:
        plt.savefig("Classical_damped.png", dpi=1000)
    plt.show()


if Error_N:
    solves = np.vectorize(solve)
    Ns = np.arange(2, 100)
    RMSEs = solves(Ns, x_y=False)

    mask = RMSEs > 1e-18
    Ns = Ns[mask]
    RMSEs = RMSEs[mask]

    plt.figure()
    plt.semilogy()

    plt.scatter(Ns, RMSEs, color="red", alpha=0.5)

    wu = 22
    fp_err = 48

    def exp_fit(x, a, b):
        return a * np.exp(-b * x)

    def exp_fit_log(x, a, b):
        return np.log(a) - b * x

    a_fit, b_fit = curve_fit(exp_fit_log, Ns[wu:fp_err], np.log(RMSEs[wu:fp_err]))[0]

    print("Decay = " + str(100 - 100*np.exp(-b_fit)) + "%")

    print("Error half life = " + str(np.log(2) / b_fit))
    print("Decay = " + str(100 - 100 * np.exp(-b_fit)) + "%")

    N_exp = np.array([Ns[wu], Ns[fp_err]])
    label_exp = str(sf(a_fit)) + r" $exp($" + str(-sf(b_fit)) + r"$N)$"

    plt.plot(
        N_exp,
        exp_fit(N_exp, a_fit, b_fit),
        label=label_exp,
        linestyle="--",
        color="black",
    )

    a_round, b_round = curve_fit(exp_fit_log, Ns[fp_err:], np.log(RMSEs[fp_err:]))[0]

    N_round = np.array([Ns[fp_err], Ns[-1]])
    label_round = str(sf(a_round)) + r" $exp($" + str(-sf(b_round)) + r"$N)$"

    plt.plot(
        N_round,
        exp_fit(N_round, a_round, b_round),
        label=label_round,
        linestyle=":",
        color="black",
    )

    plt.xlabel(r"$N$")
    plt.ylabel("RMSE")
    plt.title(r"Error in $u_N$ approximation of classical damped pendulum")
    plt.legend()

    plt.savefig("Classical_damped_err.png")
    plt.show()
