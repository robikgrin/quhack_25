from numba import jit
import numpy as np
import scipy.sparse as sp

def ising_coefs(Q: sp.csr_matrix):
    """Compute Ising coefficients J (pairwise) and h (local fields) from QUBO matrix Q.

    For MAXCUT (Q is symmetric)

    Return:
      J : sparse matrix with zeros on diagonal (same shape as Q)
      h : 1D numpy array length n
    """
    if not sp.isspmatrix_csr(Q):
        Q = sp.csr_matrix(Q)
        
    J = Q.tolil()
    J.setdiag(0)
    J = J.tocsr()
    J.data *= 0.25

    L = Q.tolil()
    L.setdiag(0)
    diag = Q.diagonal()
    h = 1/2 * L.toarray().sum(axis=1).flatten() + 1/2 * diag
    return J, h

def sigmoid_pump(step, max_steps, p_start, p_end, k):
  t = step / max_steps
  return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))

def solve(matrix):
    """
    SimCIM (coherent ising machine)
    """
    n = matrix.shape[0]
    Q_sym = sp.csr_matrix(matrix, dtype=np.float64)
    J, h = ising_coefs(Q_sym)
    T = 30
    dt = 0.001

    max_steps = int(T//dt)
                  
    p_start = 0.2
    p_end = 1.9    
    k =  6.002499433118096

    a = 1e-4 * np.random.randn(n).astype(np.float64)

    for step in range(max_steps):
        p = sigmoid_pump(step, max_steps, p_start, p_end, k)

        coupling = J.dot(a).astype(np.float64)
        
        nonlinearity = a**3
        # noise = 1e-4 * np.random.randn(n).astype(np.float64)
        da_dt = (p-1.0) * a - nonlinearity + coupling + h

        a += dt * da_dt

        a = np.clip(a, -50.0, 50.0)

    ### Бинаризация ###
    s = np.sign(a)
    s = np.where(s == 0, 1, s)  # 0 -> +1

    x = ((1 + s)//2).astype(int)
    return x 