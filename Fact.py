import numpy as np

primes = np.array([2])
def find_primes(N, n=3):
    while n<N:
        if n == N - 1 or n == N - 2:
            return primes
        elif np.sum(n % primes == 0) == 0:
            np.append(primes, n)
            n += 2
            find_primes(N, n)
        else:
            n += 2
            find_primes(N, n)

primes = find_primes(20)

print(primes)



def factorise(N, n=2, output=None):
    if output is None:
        output = []
    if n == N:
        output.append(n)
        return output
    elif N % n == 0:
        output.append(n)
        return factorise(N // n, 2, output)
    else:
        return factorise(N, n + 1, output)

# Example usage:
N = 280
factors = factorise(N)
# print(factors)  # Output: [2, 2, 7]

def uhm(limit, alpha):
    k = np.arange(limit + 1)

    a = np.empty(limit + 1)
    a[0] = 1
    a[1:] = 1 - (alpha + 1) / k[1:]
    a = np.cumprod(a)
    return a

n = int(1e6)
print(uhm(n, 1.1))


