import numpy as np
from scipy.special import gammaln

def D(N, nu, L=4*np.pi, D1_mat=None):
    """
    Build the (fractional) Chebyshev differentiation matrix of order `nu`.

    Parameters
    ----------
    N : int
        Polynomial degree; matrix is (N+1) x (N+1).
    nu : int or float
        Order of derivative. If integer, uses matrix power of D1.
    L : float, optional
        Domain length scale (appears in your formula), default 1.0.
    D1_mat : np.ndarray, optional
        First-order Chebyshev differentiation matrix of shape (N+1, N+1).
        If None and nu is integer, a global/function `D_1(N)` must exist.

    Returns
    -------
    Dmat : (N+1, N+1) ndarray
        The differentiation matrix of order `nu`.
    """
    # --- Integer case: fast and simple
    if isinstance(nu, (int, np.integer)):
        if nu == 0:
            return np.eye(N + 1)
        if D1_mat is None:
            # Expecting a user-defined D_1(N) out of scope here.
            D1_mat = D_1(N)  # noqa: F821 (assumed available in caller’s env)
        return np.linalg.matrix_power(D1_mat, int(nu))

    # --- Non-integer case
    LB = int(np.ceil(nu))

    # Indices, shapes, and base arrays
    i = np.arange(N + 1, dtype=float)[:, None]  # (N+1, 1)
    j = np.arange(N + 1, dtype=float)[None, :]  # (1, N+1)

    Dmat = np.zeros((N + 1, N + 1), dtype=float)

    # eps_j: first column factor = 2, else 1
    eps_j = np.ones((1, N + 1), dtype=float)
    eps_j[:, 0] = 2.0

    # Coefficient (constant over k)
    coeff = 2.0 * i / (eps_j * (L ** nu))

    # Base sign dependent on (i - LB) parity, as in your code
    sign_base = np.where(((i - LB) % 2) == 0, 1.0, -1.0)

    # Precompute terms depending only on k
    ks = np.arange(LB, N + 1, dtype=float)  # LB..N
    g_k_nu_half = gammaln(ks - nu + 0.5)      # shape (N-LB+1,)
    g_k_half    = gammaln(ks + 0.5)           # shape (N-LB+1,)

    # Initialize the "term" for k = LB: term = Π_{m=1..j} ((a + m - 1)/(a - m))
    # We build it once in log-space for numerical stability.
    a0 = LB - nu + 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        factors = (a0 + j - 1.0) / (a0 - j)   # (1, N+1)
        factors[:, 0] = 1.0
        # magnitude in log-space + explicit sign tracking
        log_term = np.cumsum(np.log(np.abs(factors)), axis=1)  # (1, N+1)
        term_sign = np.cumprod(np.sign(factors), axis=1)       # (1, N+1)
        term = np.exp(log_term) * term_sign                    # (1, N+1)
        term = np.where(np.isfinite(term), term, np.inf)

    # Traverse k = LB..N
    for t, k in enumerate(range(LB, N + 1)):
        rows = slice(k, N + 1)  # Only i >= k is valid
        ii = i[rows]            # (N+1-k, 1)

        a = k - nu + 1.0

        # log of the k,i-dependent ratio
        # num: Γ(i+k) Γ(k - nu + 0.5)
        # den: Γ(k + 0.5) Γ(i - k + 1) Γ(a)^2
        with np.errstate(divide='ignore', invalid='ignore'):
            log_num = gammaln(ii + k) + g_k_nu_half[t]
            log_den = g_k_half[t] + gammaln(ii - k + 1.0) + 2.0 * gammaln(a)
            num_den = np.exp(log_num - log_den)

        # alternating sign per k (the mutate-in-place from your function had no effect)
        alt_k = 1.0 if ((k - LB) % 2 == 0) else -1.0

        iter_block = num_den * coeff[rows, :] * (sign_base[rows] * alt_k) / term
        iter_block = np.where(np.isfinite(iter_block), iter_block, 0.0)
        Dmat[rows, :] += iter_block

        # Update "term" for k -> k+1 (a -> a+1) using a stable recurrence to avoid recomputing cumprod
        if k < N:
            with np.errstate(divide='ignore', invalid='ignore'):
                upd = (a + j) / (a + 1.0 - j)  # (1, N+1)
            upd[:, 0] = 1.0
            term *= np.where(np.isfinite(upd), upd, 1.0)
            term = np.where(np.isfinite(term), term, np.inf)

    return Dmat

print(D(N=5, nu=1))