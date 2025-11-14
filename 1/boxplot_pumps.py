import numpy as np
import matplotlib.pyplot as plt

from graph_generator import generate_erdos_renyi_qubo
from solvers import solve_pump_tuning, solve_qubo_neal, solve_qubo_brute_force

### Планировщики накачки ###
def linear_pump(step, max_steps, p_start, p_end):
    return p_start + (p_end - p_start) * (step / max_steps)

def exponential_pump(step, max_steps, p_start, p_end, alpha=1.0):
    return p_start + (p_end - p_start) * (1 - np.exp(-alpha * step / max_steps))

def quadratic_pump(step, max_steps, p_start, p_end):
    t = step / max_steps
    return p_start + (p_end - p_start) * t**2

def sigmoid_pump(step, max_steps, p_start, p_end, k=6):
    t = step / max_steps
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))
#####################

def relative_error(Q, x_simcim, E_opt):
    E_simcim = x_simcim @ Q @ x_simcim
    eps = 1e-12
    return abs(E_simcim - E_opt) / (abs(E_opt) + eps) * 100

def generate_qubo_batch(batch_size=20, N=10, seed=42):
    np.random.seed(seed)
    Q_list = []
    E_opt_list = []
    for i in range(batch_size):
        size = np.random.randint(N, N+1)
        prob = np.random.rand()
        Q, _= generate_erdos_renyi_qubo(size, edge_prob=prob, seed=seed + i)
        Q_list.append(Q)

        if size <= 20:
            _, E_opt = solve_qubo_brute_force(Q.toarray())
        else:
            _, E_opt = solve_qubo_neal(Q, num_reads=500)
        E_opt_list.append(E_opt)
        print(f"Сгенерирован QUBO #{i+1}, размер = {size}, вероятность = {prob}, E_opt = {E_opt:.3f}")
    return Q_list, E_opt_list

def evaluate_solver_on_batch(Q_list, E_opt_list, solve_func, solver_name=""):
    rel_errors = []
    for Q, E_opt in zip(Q_list, E_opt_list):
        x_sol = solve_func(Q)
        err = relative_error(Q, x_sol, E_opt)
        rel_errors.append(err)
    return np.array(rel_errors)

if __name__ == "__main__":
    print("Генерация батча QUBO...")
    Q_list, E_opt_list = generate_qubo_batch(batch_size=10, N=10, seed=42)

    # Определяем все pump-функции
    pump_configs = [
        ("linear", linear_pump, {}),
        ("exponential", exponential_pump, {"alpha": 2.0}),
        ("quadratic", quadratic_pump, {}),
        ("sigmoid", sigmoid_pump, {"k": 6.002499433118096})
    ]

    all_errors = {}
    labels = []

    # Оцениваем каждый планировщик
    for name, pump_func, params in pump_configs:
        def solver(Q, pf=pump_func, pp=params):
            return solve_pump_tuning(Q, pf, pp)
        errors = evaluate_solver_on_batch(Q_list, E_opt_list, solver, name)
        all_errors[name] = errors
        labels.append(name)

    # Визуализация
    plt.figure(figsize=(12, 7))
    data = [all_errors[label] for label in labels]
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5)  # ← жирная чёрная медиана
    )
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#D6843D']  # синий, зелёный, красный, фиолетовый, охра

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.ylabel('Относительная ошибка (%)', fontsize=12)
    plt.title('Сравнение планировщиков накачки в SimCIM\nЗа идеал принят dwave-neal', fontsize=14)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("pump_comparison_boxplot.png", dpi=400)
    plt.show()

    # Сводка по медианам
    print("\n\t  ======= СВОДКА МЕДИАННЫХ ОТНОСИТЕЛЬНЫХ ОШИБОК =======")
    for label in labels:
        med = np.median(all_errors[label])
        mean = np.mean(all_errors[label])
        print(f"{label:>20}: медиана = {med:.3f}%, среднее = {mean:.3f}%")