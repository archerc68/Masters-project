import numpy as np
from scipy.special import gamma, factorial
import time
from numba import njit


# --- Original triple-loop version ---
def D_loop(N, nu, L=1.0):
    LB = int(np.ceil(nu))

    def eps(j):
        return 2 if j == 0 else 1

    D_matrix = np.zeros((N + 1, N + 1))
    for i in range(LB, N + 1):
        for k in range(LB, i + 1):
            num = (-1) ** (i - k) * 2 * i * factorial(i + k - 1) * gamma(k - nu + 0.5)
            for j in range(N + 1):
                den = (
                    eps(j)
                    * (L**nu)
                    * gamma(k + 0.5)
                    * factorial(i - k)
                    * gamma(k - j - nu + 1)
                    * gamma(k + j - nu + 1)
                )
                D_matrix[i, j] += num / den
    return D_matrix


# --- First vectorized version (with mask) ---
def D_vec(N, nu, L=1.0):
    LB = int(np.ceil(nu))

    j = np.arange(N + 1)
    eps = np.ones_like(j, dtype=float)
    eps[0] = 2.0

    i = np.arange(LB, N + 1)[:, None]
    k = np.arange(LB, N + 1)[None, :]

    mask = k <= i

    num = (
        np.where((i - k) % 2 == 0, 1, -1)
        * (2 * i)
        * factorial(i + k - 1)
        * gamma(k - nu + 0.5)
    )

    denom_base = gamma(k + 0.5) * (L**nu)

    fact_ik = factorial(i - k, exact=False)
    fact_ik[~mask] = np.inf

    j_arr = j[None, None, :]
    den = (
        eps[j_arr]
        * denom_base[..., None]
        * fact_ik[..., None]
        * gamma(k[..., None] - j_arr - nu + 1)
        * gamma(k[..., None] + j_arr - nu + 1)
    )

    contrib = num[..., None] / den
    D_matrix = np.zeros((N + 1, N + 1))
    D_matrix[LB:, :] = np.sum(contrib, axis=1)

    return D_matrix


# --- Optimized meshgrid version ---
def D_matrix(N, nu, L=1.0):
    i = np.arange(N + 1)[:, None]   # column vector
    j = np.arange(N + 1)[None, :]   # row vector
    eps_j = np.where(j == 0, 2, 1)

    k_min = int(np.ceil(nu))
    D = np.zeros((N+1, N+1))

    for k in range(k_min, N+1):
        # broadcast over i,j
        sign = np.where(((i - k) % 2) == 0, 1.0, -1.0)
        num = sign * (2 * i) * factorial(i + k - 1) * gamma(k - nu + 0.5)

        den = (
            eps_j * (L**nu) * gamma(k + 0.5) *
            gamma(i - k + 1) * gamma(k - nu - j + 1) * gamma(k + j - nu + 1)
        )

        term = np.where(k <= i, num / den, 0.0)
        D += term   # accumulate over k

    return D


# --- Benchmark runner ---
def benchmark(funcs, Ns, nu=1.5):
    results = {}
    for f in funcs:
        times = []
        for N in Ns:
            start = time.perf_counter()
            f(N, nu)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results[f.__name__] = times
    return results


# Run benchmarks
Ns = [10, 20, 40, 80]
funcs = [D_loop, D_vec, D_matrix]
results = benchmark(funcs, Ns)

# Print results
print("N values:", Ns)
for name, times in results.items():
    print(f"{name:10s}:", ["{:.4f}s".format(t) for t in times])
