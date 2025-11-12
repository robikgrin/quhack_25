import numpy as np
import optuna
import scipy.sparse as sp
from dimod import BinaryQuadraticModel
import neal

from QUBO_transformers import *
from graph_generator import *

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

def solve_simcim(Q, T, dt, p_start, p_end):
    n = Q.shape[0]
    J, h = ising_coefs(Q)

    a = 1e-4 * np.random.randn(n)
    max_steps = int(T//dt)
    
    for step in range(max_steps):
        p = p_start + (p_end - p_start) * (step / max_steps)

        coupling = J.dot(a).astype(np.float64)

        nonlinearity = a**3

        da_dt = (p-1) * a - nonlinearity + coupling + h

        a += dt * da_dt

        a = np.clip(a, -100.0, 100.0)

    s = np.sign(a)
    s = np.where(s == 0, 1, s)
    x = ((1 + s) / 2).astype(int)
    return x

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim.T @ Q @ x_simcim
    print(E_simcim, E_opt)
    eps = 1e-8
    return (abs(E_simcim - E_opt)/abs(E_opt + eps)) * 100

def evaluate_params_on_batch(Q_list, E_opt_list, T, dt, p_start, p_end):
    rel_errors = []
    for Q, E_opt in zip(Q_list, E_opt_list):
        x_sim = solve_simcim(Q, T, dt, p_start, p_end)
        err = relative_error(Q, x_sim, E_opt)
        rel_errors.append(err)
    return np.mean(rel_errors)

def objective(trial, Q_list, E_opt_list):
    T = trial.suggest_float('T', 10, 50)
    dt = trial.suggest_float('dt', 1e-4, 1e-1, log=True)
    p_start = trial.suggest_float('p_start', 0.0, 0.5)
    p_end = trial.suggest_float('p_end', 1.0, 10.0)
    return evaluate_params_on_batch(Q_list, E_opt_list, T, dt, p_start, p_end)

def generate_qubo_batch(batch_size=20, N=10, seed=42):
    Q_list = []
    E_opt_list = []
    for _ in range(batch_size):
        size = np.random.randint(N, N+10)
        numbers = np.arange(size%2+1, 100, 2)
        degree = np.random.choice(numbers, size=1, replace=True)[0]

        print(size, degree)
        Q = generate_random_regular_graph_qubo(size, degree = degree, seed = seed, vis = False)
        Q_list.append(sp.csr_matrix(Q, dtype=float))
        if N <= 20:
            _, E_opt = solve_qubo_brute_force(Q)
        else:
            _, E_opt = solve_qubo_neal(Q, num_reads=500)
        E_opt_list.append(E_opt)
    return Q_list, E_opt_list

def tune_simcim_on_batch(N=10, batch_size=20, n_trials=100):
    Q_list, E_opt_list = generate_qubo_batch(batch_size=batch_size, N=N)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, Q_list, E_opt_list), n_trials=n_trials)
    return study.best_params, study.best_value

# Пример запуска
if __name__ == "__main__":
    best_params, best_error = tune_simcim_on_batch(N=180, batch_size=100, n_trials=1000)
    print("Лучшие параметры:", best_params)
    print("Средняя относительная ошибка:", best_error)