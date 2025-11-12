import numpy as np
from scipy.sparse import csr_matrix

from QUBO_transformers import *

def sigmoid_pump(step, max_steps, p_start, p_end, k=6):
    t = step / max_steps
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))


def solve(matrix, sigma=0.01):
    """
    SimCIM (coherent ising machine)
    """
    n = matrix.shape[0]
    Q_sym = csr_matrix(matrix, dtype=np.float64)
    J, h = ising_coefs(Q_sym)
    
    max_steps = 30000

    dt = 0.001              

    p_start =  0.2

    p_end = 1.9   

    k =  6.002499433118096

    a = 1e-4 * np.random.randn(n).astype(np.float64)
    
    for step in range(max_steps):
        # p = p_start + (p_end - p_start) * (step / max_steps)
        p = sigmoid_pump(step, max_steps, p_start, p_end, k)

        coupling = J @ a

        nonlinearity = a**3

        da_dt = (p-1) * a - nonlinearity + coupling + h

        # noise = sigma * np.random.randn(n).astype(np.float64)
        # a += dt * da_dt + dt * np.sqrt(np.abs(a)) * noise
        a += dt * da_dt

        a = np.clip(a, -100.0, 100.0)
    
    # Бинаризация
    s = np.sign(a)
    s = np.where(s == 0, 1, s)  # 0 -> +1

    x = ((1 + s)//2).astype(int)
    return x 
