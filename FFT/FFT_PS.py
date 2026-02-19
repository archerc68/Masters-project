import numpy as np
from scipy.fftpack import dct
from numpy.polynomial.chebyshev import chebval

N = 10
X = -np.cos(np.arange(N + 1) * np.pi / N)


def tocheb(A, X, N, B):
    # Compute initial B values
    dct(A, type=1)
    
    # Scaling
    B1 = 0.5 / N
    B[0] *= B1
    B[N] *= B1
    
    B1 = 2.0 * B1  # Now B1 = 1/N
    for i in range(1, N):
        B[i] *= B1


def diffcheb(A):

    N = len(A) - 1
    B = np.zeros_like(A)
    
    if N < 2:
        return B  # Derivative of constant or linear is trivial
    
    # Initialize
    A1 = A[N]
    A2 = A[N-1]
    B[N] = 0.0
    B[N-1] = 2.0 * N * A1
    
    # Next term
    A1 = A2
    A2 = A[N-2]
    B[N-2] = 2.0 * (N-1) * A1
    
    # Recurrence
    for i in range(N-2, 1, -1):
        A1 = A2
        A2 = A[i-1]
        B[i-1] = B[i+1] + 2.0 * i * A1
    
    # Final term
    B[0] = 0.5 * B[2] + A2
    
    return B


def fromcheb(A, X):

    N = len(A) - 1
    Bcoeff = np.zeros_like(A)
    
    # First and last coefficients
    Bcoeff[0] = A[0]
    Bcoeff[N] = A[N]
    
    # Alternating sign scaling for intermediate coefficients
    sign = 0.5
    for i in range(1, N):
        sign = -sign
        Bcoeff[i] = sign * A[i]
    
    # Evaluate Chebyshev series at nodes
    B = chebval(X, Bcoeff)
    return B

tocheb(A,X,N,B)
diffcheb(A)
fromcheb(A,X)