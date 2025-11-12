import numpy as np
import scipy.sparse as sp

def sigmaz_k(k: int, n: int) -> sp.csr_matrix:
    if k < 0 or k >= n:
        raise ValueError("k out of range")
    left_part = sp.eye(2 ** k, dtype=np.complex64)
    right_part = sp.eye(2 ** (n - 1 - k), dtype=np.complex64)
    Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex64)
    return sp.kron(sp.kron(left_part, Z, format='csr'), right_part, format='csr')


def ising_coefs(Q: sp.csr_matrix):
    """Compute Ising coefficients J (pairwise) and h (local fields) from QUBO matrix Q.

    For MAXCUT (Q is symmetric)

    Return:
      J : sparse matrix with zeros on diagonal (same shape as Q)
      h : 1D numpy array length n
    """
    if not sp.isspmatrix_csr(Q):
        Q = sp.csr_matrix(Q)

    J = Q.copy()
    J.setdiag(0)
    J.data = 0.25 * J.data

    L = Q.copy()
    L.setdiag(0)
    diag = Q.diagonal()
    h = 1/2 * L.toarray().sum(axis=1).flatten() + 1/2 * diag
    return J, h


def ising(J: sp.csr_matrix, h: np.array, n: int) -> sp.csr_matrix:
    """Build full Ising Hamiltonian as sparse matrix of size 2^n x 2^n.

    Hamiltonian used:
        H = sum_{i<j} J_{ij} sigma_z^i sigma_z^j + sum_i h_i sigma_z^i

    Note: sigma_z eigenvalues are +1 (|0>) and -1 (|1>).
    """
    dim = 2 ** n
    res = sp.csr_matrix((dim, dim), dtype=np.complex64)

    rows, cols = J.nonzero()
    for i, j in zip(rows, cols):
        term = J[i, j] * (sigmaz_k(i, n) @ sigmaz_k(j, n))
        res = res + term

    for i in range(n):
        res = res + h[i] * sigmaz_k(i, n)

    return res


def qubo_to_ising(Q: sp.csr_matrix) -> sp.csr_matrix:
    """Convert QUBO matrix Q into Ising Hamiltonian sparse matrix.

    Returns full Hamiltonian (2^n x 2^n).
    Also returns (J,h) if needed by caller.
    """
    if not sp.isspmatrix_csr(Q):
        Q = sp.csr_matrix(Q)
    n = Q.shape[0]
    J, h = ising_coefs(Q)
    H = ising(J, h, n)
    return H


