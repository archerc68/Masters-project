import numpy as np
import matplotlib.pyplot as plt


def fractional_weight_matrix(powers, N):
    powers = np.array(powers)
    K = np.arange(1, N)
    W = np.empty((len(powers), N))
    W[:, 0] = 1.0
    W[:, 1:] = np.cumprod((powers[:, None] - (K - 1)) / K, axis=1)
    W *= (-1) ** np.arange(N)
    return W


def fractional_linear_solver(coeff, powers, f, y0, T, h):
    N = int(T / h)
    y = np.zeros(N + 1)
    y[0] = y0
    t = np.linspace(0, T, N + 1)

    # Precompute weights matrix: shape (len(powers), N+1)
    weights = fractional_weight_matrix(powers, N + 1)

    # Precompute index matrix for y[n - k]
    history = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        k_max = n + 1
        history[n, :k_max] = y[n - np.arange(k_max)]

    # Main loop
    for n in range(1, N + 1):
        frac_terms = np.dot(
            coeff,
            [
                np.dot(weights[i, : n + 1], history[n, : n + 1])
                for i in range(len(coeff))
            ],
        )
        y[n] = f(y[n - 1]) - frac_terms

    return t, y


# Example usage
if __name__ == "__main__":
    coeff = [1.0, 1.0]  # Coefficients for y and D^0.5 y
    powers = [0.0, 0.5]  # Orders: y^0 and D^0.5 y

    def f(y):
        return np.sin(y)  # Right-hand side function

    y0 = 10  # Initial condition
    T = 10  # Total time
    h = 0.01  # Time step

    t, y = fractional_linear_solver(coeff, powers, f, y0, T, h)

    # Plot the result
    plt.plot(t, y)
    plt.xlabel("Time t")
    plt.ylabel("y(t)")
    plt.title("Fractional Linear System Response")
    plt.grid(True)
    plt.show()
