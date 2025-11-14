import numpy as np
import time

from solution_sim import solve
from graph_generator import generate_erdos_renyi_qubo
from solvers import solve_qubo_neal, solve_qubo_brute_force

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim.T @ Q @ x_simcim
    eps = 1e-8
    return abs(E_simcim - E_opt) / abs(E_opt + eps) * 100, E_simcim

def generate_qubo_batch(batch_size=20, N=10, seed=42):
    np.random.seed(seed)
    Q_list = []
    E_opt_list = []
    for i in range(batch_size):
        size = np.random.randint(N, 5*N)
        prob = np.random.rand()
        Q, _= generate_erdos_renyi_qubo(size, edge_prob=prob, seed=seed + i)
        Q_list.append(Q)

        if size <= 20:
            _, E_opt = solve_qubo_brute_force(Q.toarray())
        else:
            _, E_opt = solve_qubo_neal(Q, num_reads=500)
        E_opt_list.append(E_opt)
        print(f"Сгенерирован QUBO #{i+1}, размер = {size}, вероятность ребра = {prob:.4f}, E_opt = {E_opt:.3f}")
    return Q_list, E_opt_list

if __name__ == '__main__':
    Q_list, E_opt_list = generate_qubo_batch(N = 300, batch_size=1, seed = 42)

    print('\nНачало тестирования SimCIM на сгенерированных QUBO...')
    errors = []
    i = 1
    for Q,E_opt in zip(Q_list, E_opt_list):
        start = time.time()
        x = solve(Q)
        finish = time.time()
        print(f'Время вычислений: {finish - start}')
        err, E_sim = relative_error(Q, x, E_opt)
        errors.append(err)
        print(f"Граф #{i}: SimCIM: {E_sim}, Dwave: {E_opt} \tОтносительная ошибка: {errors[-1]:.3f}%")
        i+=1
    print(f"\nСредняя относительная ошибка SimCIM: [{np.mean(errors):.3f} +- {np.std(errors)/np.sqrt(len(errors) - 1):.3f}]%")