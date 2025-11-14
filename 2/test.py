import numpy as np
import time

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from solution_rwa import solve
from oriented_graph_generator import visualize_directed_graph_with_queries_colored_by_wave


def calculate_scores_with_ideal_reference(team_results, test_graphs, test_requests, num_wavelengths):
    """
    Вычисляет метрики r_score и e_score, используя ИДЕАЛЬНЫЙ эталон:
    - r̂_i = R_i (все запросы могут быть удовлетворены).
    - ê_i = R_i (минимальное количество рёбер = количество запросов, если все пути длины 1).

    Это гарантирует: r_score <= 1 и e_score >= 1.
    
    Аргументы:
        team_results (list of dict): [{'satisfied_requests': r_i, 'used_edges': e_i}, ...]
        test_graphs (list of dict): [{'num_vertices': V_i, 'num_edges': E_i}, ...]
        test_requests (list of dict): [{'num_requests': R_i}, ...]
        num_wavelengths (list) : 

    Возвращает:
        tuple: (r_score, e_score), где 0 <= r_score <= 1, e_score >= 1.
    """
    n_tests = len(team_results)
    
    # --- Шаг 1: Определение эталонных значений ---
    reference_results = []
    for i in range(n_tests):
        R_i = test_requests[i]['num_requests']
        # Идеальный эталон: удовлетворить все запросы
        r_hat_i = float(R_i) * 1.1
        # Идеальный эталон по рёбрам: минимум R_i (по одному ребру на запрос)
        e_hat_i = float(R_i)*3 if R_i > 0 else 0.0
        reference_results.append({'satisfied_requests': r_hat_i, 'used_edges': e_hat_i})

    # --- Шаг 2: Вычисление весов тестов W_i ---
    W_values = []
    for i in range(n_tests):
        V_i = test_graphs[i]['num_vertices']
        E_i = test_graphs[i]['num_edges']
        R_i = test_requests[i]['num_requests']
        if E_i == 0:
            raise ValueError(f"Тест {i}: E_i=0, деление на ноль.")
        W_i = (V_i * R_i) / (num_wavelengths[i] * E_i)
        W_values.append(W_i)

    # Нормализация весов
    W_min, W_max = min(W_values), max(W_values)
    epsilon = 1e-9
    if W_max == W_min:
        w_values = [1.0] * n_tests
    else:
        w_values = [(W_i - W_min) / (W_max - W_min) + epsilon for W_i in W_values]
    W_total = sum(w_values)

    # --- Шаг 3: Вычисление итоговых метрик ---
    r_numerator = e_numerator = 0.0
    for i in range(n_tests):
        r_i = team_results[i]['satisfied_requests']
        e_i = team_results[i]['used_edges']
        r_hat_i = reference_results[i]['satisfied_requests']
        e_hat_i = reference_results[i]['used_edges']
        w_i = w_values[i]

        # Обработка теста без запросов
        if test_requests[i]['num_requests'] == 0:
            ratio_r = 1.0 if r_i == 0 else 0.0
            ratio_e = 1.0 if e_i == 0 else float('inf')
        else:
            ratio_r = r_i / r_hat_i
            ratio_e = e_i / e_hat_i

        r_numerator += w_i * ratio_r
        e_numerator += w_i * ratio_e

    r_score = r_numerator / W_total
    e_score = e_numerator / W_total 

    return r_score, e_score

def visualize_directed_graph_with_solutions_colored_by_wavelength(graph_str, solutions, title="Направленный граф с решениями по длинам волн"):
    """
    Визуализирует направленный граф из строки, выделяя пути решений разными цветами по длинам волн.
    Каждый путь (primary/backup) для запроса отображается отдельно, если он дважды используется,
    он будет отображен дважды (например, primary и backup могут использовать одинаковые ребра,
    но будут отображаться разными цветами).
    
    Аргументы:
        graph_str (str): Строка с описанием графа.
        solutions (dict): Словарь решений в формате:
            {(0, 6): {'primary': {'wavelength': 0, 'path': [0, 2, 6]}, 'backup': {'wavelength': 1, 'path': [0, 9, 7, 6]}}}
        title (str): Заголовок графика.
    """
    # Парсинг graph_str
    graph_lines = graph_str.strip().split('\n')
    if len(graph_lines) < 2:
        raise ValueError("graph_str содержит недостаточно строк")

    header = graph_lines[0].split()
    if len(header) != 2:
        raise ValueError("Первая строка graph_str должна содержать два числа: число вершин и число рёбер")

    n_vertices = int(header[0])
    n_edges = int(header[1])

    # Создание графа NetworkX
    G = nx.DiGraph()
    G.add_nodes_from(range(n_vertices))

    # Парсинг рёбер
    edges = []
    for i, line in enumerate(graph_lines[1:], 1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Строка ребра {i+1} в graph_str должна содержать две вершины: 'источник цель'")
        u, v = int(parts[0]), int(parts[1])
        if u == v:
            raise ValueError(f"Петля обнаружена: {u}->{v}. Рёбра должны быть между разными вершинами.")
        if u < 0 or u >= n_vertices or v < 0 or v >= n_vertices:
            raise ValueError(f"Вершина {u} или {v} выходит за пределы диапазона [0, {n_vertices-1}]")
        edges.append((u, v))

    G.add_edges_from(edges)

    # Настройка визуализации
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42) # Увеличенный k для лучшего размещения

    # Рисуем основные элементы графа
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray', edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=30, edge_color='lightgray', width=1, connectionstyle="arc3,rad=0.1") # Слегка изогнутые ребра для читаемости
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

    # Сбор всех уникальных путей для раскраски
    all_paths_info = []
    path_counter = 0
    for (src, dst), solution_data in solutions.items():
        for path_type, path_details in solution_data.items():
            wavelength = path_details.get('wavelength')
            path = path_details.get('path')
            if path and len(path) >= 2:
                # Проверка корректности пути
                for node in path:
                    if node < 0 or node >= n_vertices:
                        raise ValueError(f"Узел {node} в пути {path} выходит за пределы графа [0, {n_vertices-1}]")
                all_paths_info.append({
                    'request': (src, dst),
                    'type': path_type,
                    'wavelength': wavelength,
                    'path': path,
                    'id': path_counter
                })
                path_counter += 1

    # Генерация цветов для каждого пути
    n_paths = len(all_paths_info)
    if n_paths > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, n_paths)) if n_paths <= 20 else plt.cm.hsv(np.linspace(0, 1, n_paths))
        # Увеличиваем насыщенность и яркость
        # Рисование путей
        edge_patches = []
        edge_labels = []
        for idx, path_info in enumerate(all_paths_info):
            path = path_info['path']
            color = colors[idx]
            request = path_info['request']
            path_type = path_info['type']
            wavelength = path_info['wavelength']

            # Рисование рёбер пути
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=3, arrowsize=30, arrowstyle='->', connectionstyle="arc3,rad=0.15") # Более изогнутые для пересекающихся путей

            # Рисование узлов пути (источник, сток, промежуточные)
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=900, node_color=color, edgecolors='black', linewidths=3)
            nx.draw_networkx_nodes(G, pos, nodelist=path[1:-1], node_size=800, node_color=color, edgecolors='black', linewidths=2)
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=900, node_color=color, edgecolors='black', linewidths=3)

            # Добавление в легенду
            label = f"Req {request[0]}->{request[1]}, {path_type}, λ={wavelength}"
            edge_patches.append(Patch(color=color, label=label))
            edge_labels.append(label)

    # Добавляем легенду
    if edge_patches:
        plt.legend(handles=edge_patches, loc='upper right', fontsize=16, framealpha=0.8)

    plt.title(title, fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    team_results = []
    graphs = []
    requests = []
    waves = []

    for i in range (1, 10):
        path = f'test_case/{i}'
        with open(f"{path}/graph.txt", 'r') as f:
            graph_str = f.read()
            v_num, r_num = list(map(int, graph_str.split()[:2]))

        with open(f"{path}/queries.txt", 'r') as f:
            queries_str = f.read()
            waves_num, req_num = list(map(int, queries_str.split()[:2]))  
        visualize_directed_graph_with_queries_colored_by_wave(graph_str, queries_str)
        graphs.append({'num_vertices': v_num, 'num_edges': r_num})
        requests.append({'num_requests': req_num})
        waves.append(waves_num)
        
        start = time.time()
        solution = solve(path + '/graph.txt', path + '/queries.txt')
        final = time.time()
       
        visualize_directed_graph_with_solutions_colored_by_wavelength(graph_str, solution, 'Пример решения одного запроса')

        print('Time calc:', final - start)
        print(solution)

        # Подсчет уникальных использованных ребер (для неориентированного графа)
        used_edges = set()

        for request_key, paths in solution.items():
            # Обработка primary пути
            if 'primary' in paths and paths['primary'] is not None:
                path = paths['primary']['path']
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i+1])))  # Сортируем для неориентированного графа
                    used_edges.add(edge)
            
            # Обработка backup пути
            if 'backup' in paths and paths['backup'] is not None:
                path = paths['backup']['path']
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i+1])))  # Сортируем для неориентированного графа
                    used_edges.add(edge)

        num_used_edges = len(used_edges)
        solved_reqs = len(solution)
        team_results.append({'satisfied_requests': solved_reqs, 'used_edges': num_used_edges})

    r, e = calculate_scores_with_ideal_reference(team_results, graphs, requests, waves)

    print(f"r_score: {r:.2f}")
    print(f"e_score: {e:.2f}")