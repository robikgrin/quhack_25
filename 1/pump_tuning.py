import numpy as np
import optuna

from solvers import solve_qubo_brute_force, solve_qubo_neal, solve_pump_tuning
from graph_generator import generate_erdos_renyi_qubo

### Планировщики накачки ###
def linear_pump(step, max_steps, p_start, p_end):
    return p_start + (p_end - p_start) * (step / max_steps)

def exponential_pump(step, max_steps, p_start, p_end, alpha=1.0):
    return p_start + (p_end - p_start) * (1 - np.exp(-alpha * step / max_steps))

def quadratic_pump(step, max_steps, p_start, p_end):
    t = step / max_steps
    return p_start + (p_end - p_start) * t**2

def sigmoid_pump(step, max_steps, p_start, p_end, k):
  t = step / max_steps
  return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))
#############################

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim.T @ Q @ x_simcim
    eps = 1e-8
    return abs(E_simcim - E_opt) / abs(E_opt + eps) * 100

def evaluate_params_on_batch(Q_list, E_opt_list, pump_func, pump_params=None):
    rel_errors = []
    for Q, E_opt in zip(Q_list, E_opt_list):
        x_sim = solve_pump_tuning(Q, pump_func, pump_params)
        err = relative_error(Q, x_sim, E_opt)
        rel_errors.append(err)
    return np.mean(rel_errors)

def objective(trial, Q_list, E_opt_list):
    pump_type = trial.suggest_categorical('pump_type', ['sigmoid'])

    pump_params = {}
    if pump_type == 'exponential':
        pump_params['alpha'] = trial.suggest_float('alpha', 0.1, 10.0)
    elif pump_type == 'sigmoid':
        pump_params['k'] = trial.suggest_float('k', 1.0, 10.0)

    # Выбор функции
    if pump_type == 'linear':
        pump_func = linear_pump
    elif pump_type == 'exponential':
        pump_func = exponential_pump
    elif pump_type == 'quadratic':
        pump_func = quadratic_pump
    elif pump_type == 'sigmoid':
        pump_func = sigmoid_pump

    return evaluate_params_on_batch(Q_list, E_opt_list, pump_func, pump_params)

def generate_qubo_batch(batch_size=20, N=10, seed=42):
    Q_list = []
    E_opt_list = []
    for _ in range(batch_size):
        size = np.random.randint(N, N+10)
        edge_prob = np.random.rand()
        Q, _ = generate_erdos_renyi_qubo(size, edge_prob=edge_prob, seed = seed)

        Q_list.append(Q)
        print(_)

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


if __name__ == "__main__":
    best_params, best_error = tune_simcim_on_batch(N=10, batch_size=5, n_trials=100)
    print("Лучшие параметры:", best_params)
    print("Средняя относительная ошибка:", best_error)