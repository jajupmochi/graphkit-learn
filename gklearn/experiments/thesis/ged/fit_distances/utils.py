import numpy as np


def vec2sym_mat(v):
    """
    Convert a vector encoding a symmetric matrix into a matrix
    See Golub and Van Loan, Matrix Computations, 3rd edition, p21
    """
    n = int((-1+np.sqrt(1+8*len(v)))/2)  # second order resolution
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # Golub van Loan, Matrix Computations, Eq. 1.2.2, p21
            M[i, j] = M[j, i] = v[i*n - (i+1)*(i)//2 + j]
    return M
