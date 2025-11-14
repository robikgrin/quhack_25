import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_erdos_renyi_qubo
from solvers import solve_qubo_with_history


def sigmoid_pump(step, max_steps, p_start, p_end, k=6):
    t = step / max_steps
    return p_start + (p_end - p_start) / (1 + np.exp(-k * (t - 0.5)))

if __name__ == "__main__":
    n = 100
    Q_csr, G = generate_erdos_renyi_qubo(n_nodes=n, edge_prob=0.05, seed=42, vis=False)

    pump_pars = {'k': 6.002}
    x, amp_history, time_points = solve_qubo_with_history(Q_csr, pump_func = sigmoid_pump, pump_params = pump_pars, record_every=5)

    maxcut_value = x @ Q_csr @ x

    print(f"Найденное значение MAXCUT: {maxcut_value:.2f}")

    plt.figure(figsize=(14, 6), dpi = 100)

    # Эволюция амплитуд
    plt.subplot(1, 2, 1)
    selected_nodes = np.random.choice(n, size=100, replace=False)
    for i in selected_nodes:
        plt.plot(time_points, amp_history[:, i], label=f'узел {i}')
    plt.xlabel('Время (усл. ед.)', fontsize=14)
    plt.ylabel('Амплитуда $a_i(t)$' ,fontsize=14)
    plt.title('Эволюция амплитуд в SimCIM', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Разрез графа
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G, seed=42, k=0.15)
    colors = ['red' if x[i] == 1 else 'blue' for i in range(n)]
    nx.draw(G, pos, node_color=colors, node_size=50, with_labels=False, width=0.5)
    plt.title(f'Разрез графа (MAXCUT = {maxcut_value:.1f}) модели Эрдёша — Реньи\n N = {n} вершин', fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    plt.show()