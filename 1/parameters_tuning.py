import numpy as np
import optuna

from graph_generator import generate_erdos_renyi_qubo
from solvers import solve_qubo_brute_force, solve_qubo_neal, solve_pars_tuning

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim.T @ Q @ x_simcim
    eps = 1e-8
    return (abs(E_simcim - E_opt)/abs(E_opt + eps)) * 100

def evaluate_params_on_batch(Q_list, E_opt_list, T, dt, p_start, p_end):
    rel_errors = []
    for Q, E_opt in zip(Q_list, E_opt_list):
        x_sim = solve_pars_tuning(Q, T, dt, p_start, p_end)
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
    np.random.seed(seed)
    Q_list = []
    E_opt_list = []
    for i in range(batch_size):
        size = np.random.randint(N, N+1)
        prob = np.random.rand()
        Q, _ = generate_erdos_renyi_qubo(size, edge_prob=prob, seed=seed + i)
        Q_list.append(Q)

        if size <= 20:
            _, E_opt = solve_qubo_brute_force(Q.toarray())
        else:
            _, E_opt = solve_qubo_neal(Q, num_reads=500)
        E_opt_list.append(E_opt)
        print(f"Сгенерирован QUBO #{i+1}, размер = {size}, вероятность = {prob}, E_opt = {E_opt:.3f}")
    return Q_list, E_opt_list

def tune_simcim_on_batch(N=10, batch_size=20, n_trials=100):
    Q_list, E_opt_list = generate_qubo_batch(batch_size=batch_size, N=N)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, Q_list, E_opt_list), n_trials=n_trials)
    return study.best_params, study.best_value

# Пример запуска
if __name__ == "__main__":
    best_params, best_error = tune_simcim_on_batch(N=10, batch_size=5, n_trials=1000)
    print("Лучшие параметры:", best_params)
    print("Средняя относительная ошибка:", best_error)