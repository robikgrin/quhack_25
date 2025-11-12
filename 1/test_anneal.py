import numpy as np
from scipy.sparse import csr_matrix
from dimod import BinaryQuadraticModel
import neal

from QUBO_transformers import *
from graph_generator import *
from ANALing import *

def solve_qubo_neal(Q, num_reads=200):
    """Надёжное приближение оптимума через Simulated Annealing."""
    n = Q.shape[0]
    Q_dict = {}
    for i in range(n):
        start = Q.indptr[i]
        end = Q.indptr[i + 1]
        for idx in range(start, end):
            j = Q.indices[idx]
            val = Q.data[idx]
            if val != 0:
                Q_dict[(i, j)] = -val

    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first
    x_opt = np.array([best.sample[i] for i in range(n)], dtype=int)
    energy = x_opt.T @ Q @ x_opt
    return x_opt, energy

def solve_qubo_brute_force(Q):
    """Точное решение для малых N (N <= 20)."""
    n = Q.shape[0]
    best_val = np.inf
    best_x = None
    for i in range(2 ** n):
        x = np.array([(i >> j) & 1 for j in range(n)], dtype=int)
        val = - x.T @ Q @ x
        if val < best_val:
            best_val = val
            best_x = x
    return best_x, abs(best_val)

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim.T @ Q @ x_simcim
    eps = 1e-8
    return abs(E_simcim - E_opt) / abs(E_opt + eps) * 100

def generate_qubo_batch(batch_size=20, N=10, seed=42):
    Q_list = []
    E_opt_list = []
    for _ in range(batch_size):
        size = np.random.randint(N, N+10)
        numbers = np.arange(size%2+1, N//3, 2)
        degree = np.random.choice(numbers, size=1, replace=True)[0]
        print(size, degree)
        Q = generate_random_regular_graph_qubo(size, degree = degree, seed = seed)

        Q_list.append(sp.csr_matrix(Q, dtype=float))
        

        if N <= 20:
            _, E_opt = solve_qubo_brute_force(Q)
        else:
            _, E_opt = solve_qubo_neal(Q, num_reads=500)
        E_opt_list.append(E_opt)
    return Q_list, E_opt_list


Q_list, E_opt_list = generate_qubo_batch(N = 180, batch_size=100)
errors = []
for Q,E_opt in zip(Q_list, E_opt_list):
    x = solve(Q)
    errors.append(relative_error(Q, x, E_opt))

print(np.mean(errors))