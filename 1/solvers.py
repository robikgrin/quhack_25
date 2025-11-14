import numpy as np

from dimod import BinaryQuadraticModel
import neal

from QUBO_transformers import ising_coefs

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


    t = step / max_steps
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))

def solve_pump_tuning(Q, pump_func, pump_params=None):
    """
    Функция для настройки планировщика накачки при лучших параметрах из `solve_pars_tuning`.
    """
    if pump_params is None:
        pump_params = {}

    n = Q.shape[0]
    J, h = ising_coefs(Q)
    
    ### ПАРАМЕТРЫ (ФИКСИРОВАННЫЕ) ###
    T = 30
    dt = 0.001
    max_steps = int(T//dt)
    p_start = 0.2
    p_end = 1.9
    ################################

    a = 1e-4 * np.random.randn(n)

    for step in range(max_steps):
        p = pump_func(step, max_steps, p_start, p_end, **pump_params)

        coupling = J.dot(a).astype(np.float64)
        nonlinearity = a**3
        da_dt = (p - 1) * a - nonlinearity + coupling + h
        a += dt * da_dt
        a = np.clip(a, -100.0, 100.0)

    s = np.sign(a)
    s = np.where(s == 0, 1, s)
    x = ((1 + s) / 2).astype(int)
    return x

def solve_pars_tuning(Q, T, dt, p_start, p_end):
    """
    Функция для настройки параметров SimCIM при линейной накачке.
    """
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

def solve_qubo_with_history(Q_csr, pump_func, record_every=10, pump_params=None):
    n = Q_csr.shape[0]
    J, h = ising_coefs(Q_csr)

    T = 30.0
    dt = 0.001
    max_steps = int(T / dt)
    p_start, p_end, k = 0.2, 1.9, 6.0
    a = 1e-4 * np.random.randn(n).astype(np.float64)
    history = []
    times = []

    for step in range(max_steps):
        p = pump_func(step, max_steps, p_start, p_end, **pump_params)
        da_dt = (p - 1.0) * a - a**3 + J.dot(a) + h
        a += dt * da_dt
        a = np.clip(a, -50, 50)

        if step % record_every == 0:
            history.append(a.copy())
            times.append(step * dt)

    history = np.array(history)
    s = np.sign(a)
    s = np.where(s == 0, 1, s)
    x = ((1 + s) // 2).astype(int)
    return x, history, np.array(times)
