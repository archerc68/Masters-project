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

assert len(alpha_j) == n & len(alpha_j) == len(d_j)


# [0, 1]
def diff_mat(alpha, N):
    D_matrix_T = np.zeros((N, N))
    k = np.arange(1, N, 2)

    for i in k:
        D_matrix_T += np.diagflat(np.arange(i, N), i)
    D_matrix = D_matrix_T.T
    D_matrix[:, 0] /= 2

    D_matrix *= 4
    return np.linalg.matrix_power(D_matrix, alpha)


def solve(N, x_y):
    Big_mat = diff_mat(2, N) + np.eye(N)
    Big_mat[:, -1] = (diff_mat(alpha_j[0], N) @ chebvander(-1, N - 1).T)[:, 0]
    Big_mat[:, -2] = (diff_mat(alpha_j[1], N) @ chebvander(-1, N - 1).T)[:, 0]

    Big_mat_inv = np.linalg.inv(Big_mat)

    column_vec = np.zeros(N)
    column_vec[-1], column_vec[-2] = d_j[0], d_j[1]

    a_i = column_vec @ Big_mat_inv

    x = np.linspace(0, 1, 250)

    def approx(x):
        ts = 2 * x - 1
        return a_i @ chebvander(ts, N - 1).T

    def approx_err(x):
        return (approx(x) - np.cos(x))**2

    mean_err = np.sqrt(quad(approx_err, 0, 1)[0])

    if x_y:
        return x, approx(x), mean_err
    else:
        return mean_err


# # U_3 plot
# x, y, mean_err = solve(5, x_y=True)


# plt.figure()
# plt.plot(x, y)
# plt.plot(x, np.cos(x))
# plt.show()

def sf(x, sig_figs):
    leading_order = 10**int(np.floor(np.log10(np.abs(x))))
    terms = np.round(x/leading_order, sig_figs)
    return terms*leading_order



# U_N plot
plt.figure()
plt.semilogy()

Ns = np.arange(2, 250)
solves = np.vectorize(solve)
mean_err_n = solves(Ns, x_y=False)


def exp_fit(x, a, b):
    return a * np.exp(-b * x)


def exp_fit_log(x, a, b):
    return np.log(a) - b * x

mask = mean_err_n > 1e-18
Ns = Ns[mask]
mean_err_n = mean_err_n[mask]


fp_err = 11
a_fit, b_fit = curve_fit(exp_fit_log, Ns[:fp_err], np.log(mean_err_n[:fp_err]))[0]

print("Error half life = " + str(np.log(2)/b_fit))
print("Decay = " + str(100 - 100*np.exp(-b_fit)) + "%")

N_exp = np.array([Ns[0], Ns[fp_err]])
label_exp = str(sf(a_fit, 2)) + r" $exp($" + str(-sf(b_fit, 2)) + r"$N)$"

plt.plot(
    N_exp, exp_fit(N_exp, a_fit, b_fit), label=label_exp, linestyle="--", color="black"
)

a_round, b_round = curve_fit(exp_fit_log, Ns[fp_err:], np.log(mean_err_n[fp_err:]))[0]

N_round = np.array([Ns[fp_err], Ns[-1]])
label_round = str(sf(a_round, 2)) + r" $exp($" + str(-sf(b_round, 2)) + r"$N)$"

plt.plot(
    N_round, exp_fit(N_round, a_round, b_round), label=label_round, linestyle=":", color="black"
)


plt.scatter(Ns, mean_err_n, color="red")

midway_exp = np.sqrt(mean_err_n[0]*mean_err_n[fp_err])
midway_round = np.sqrt(mean_err_n[fp_err]*mean_err_n[-1])

plt.text(fp_err+10, midway_exp, "Exponential convergence")
plt.text(0.5*(fp_err + Ns[-1])-15, midway_round*10**1.5, "Round off error")

plt.xlabel(r"$N$")
plt.ylabel("RMS Error")
plt.title(r"Error in $U_N$ approximation")
plt.legend()
plt.savefig("U_N_approx_cosine.png", dpi=1000)
plt.show()
