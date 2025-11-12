import numpy as np
from collections import defaultdict
import networkx as nx

import scipy.sparse as sp

from ANALing import *
from oriented_graph_generator import *

def paths_share_edge(p1, p2):
    """Проверяет, имеют ли два пути общие ребра"""
    return bool(set(zip(p1, p1[1:])) & set(zip(p2, p2[1:])))

def available(path, wl, used):
    """Проверяет, доступен ли путь на указанной длине волны"""
    for e in zip(path, path[1:]):
        if used[e][wl]:
            return False
    return True

def solve(graph_file: str, requests_file: str):
    # 1. Считываем граф
    with open(graph_file) as f:
        lines = f.read().strip().splitlines()
    n, m = map(int, lines[0].split())
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(1, m + 1):
        u, v = map(int, lines[i].split())
        G.add_edge(u, v)

    # 2. Считываем запросы
    with open(requests_file) as f:
        lines = f.read().strip().splitlines()
    Lambda = int(lines[0])
    R = int(lines[1])
    requests = [tuple(map(int, lines[2 + i].split())) for i in range(R)]

    # 3. Генерируем возможные пары путей для каждого запроса
    options = {}
    max_path_length = 0
    
    for s, t in requests:
        if not nx.has_path(G, s, t):
            options[(s, t)] = []
            continue
        pairs = []
        try:
            all_paths = list(nx.shortest_simple_paths(G, s, t))
            for p1 in all_paths:
                for p2 in all_paths:
                    if p1 != p2 and not paths_share_edge(p1, p2):
                        pairs.append((p1, p2))
                        max_path_length = max(max_path_length, len(p1) + len(p2))
                        if len(pairs) >= 3:
                            break
                if len(pairs) >= 3:
                    break
        except Exception as e:
            options[(s, t)] = []
            continue
        options[(s, t)] = pairs

    # 4. Сортируем запросы по количеству вариантов
    requests_sorted = sorted(requests, key=lambda r: len(options[r]))

    # 5. Инициализируем состояние использования
    used = defaultdict(lambda: [False] * Lambda)

    # 6. Вычисляем параметры для QUBO согласно статье
    M = max_path_length
    alpha = 1.0
    beta = R * (M - 2) + 3  # Согласно условию beta/alpha > |R|(M-2) + 2
    rho = beta * (R + 1) + 100  # Согласно условию для коэффициента штрафа

    # 7. Основной цикл обработки запросов
    result = {}

    for s, t in requests_sorted:
        candidates = []
        for p1, p2 in options[(s, t)]:
            for l1 in range(Lambda):
                for l2 in range(Lambda):
                    if l1 != l2 and available(p1, l1, used) and available(p2, l2, used):
                        candidates.append((p1, p2, l1, l2))
        
        if not candidates:
            continue

        # 8. Формируем QUBO-матрицу в формате CSR
        n = len(candidates)
        
        # Подготовка данных для CSR-матрицы
        row_indices = []
        col_indices = []
        data = []
        
        # Линейные члены (диагональ)
        for i in range(n):
            p1, p2, _, _ = candidates[i]
            # Минимизируем использование ресурсов (alpha * длина путей)
            # и максимизируем количество удовлетворенных запросов (-beta)
            cost = alpha * ((len(p1) - 1) + (len(p2) - 1)) - beta
            row_indices.append(i)
            col_indices.append(i)
            data.append(cost)
        
        # Квадратичные члены (штрафы за конфликты)
        for i in range(n):
            for j in range(i+1, n):
                p1_i, p2_i, l1_i, l2_i = candidates[i]
                p1_j, p2_j, l1_j, l2_j = candidates[j]
                
                conflict = False
                
                # Проверка конфликта между рабочими путями
                if l1_i == l1_j and paths_share_edge(p1_i, p1_j):
                    conflict = True
                
                # Проверка конфликта между резервными путями
                if l2_i == l2_j and paths_share_edge(p2_i, p2_j):
                    conflict = True
                
                # Проверка конфликта между рабочим и резервным путями разных запросов
                if l1_i == l2_j and paths_share_edge(p1_i, p2_j):
                    conflict = True
                if l2_i == l1_j and paths_share_edge(p2_i, p1_j):
                    conflict = True
                
                # Если есть конфликт, добавляем штраф
                if conflict:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(rho)
                    
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(rho)

        # Создаем CSR-матрицу
        Q = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

        # 9. Решаем QUBO с помощью solve_anal (уже реализован пользователем)
        solution = solve_anal(Q)
        
        # 10. Выбираем оптимальное решение
        chosen = None
        for i in range(len(solution)):
            if solution[i] == 1:
                chosen = i
                break
        
        if chosen is None:
            # Если нет решений, выбираем кратчайшую пару
            costs = [len(c[0]) + len(c[1]) for c in candidates]
            chosen = int(np.argmin(costs))
        
        if chosen is None:
            continue
        
        p1, p2, l1, l2 = candidates[chosen]
        
        # 11. Обновляем состояние использования
        for e in zip(p1, p1[1:]):
            used[e][l1] = True
        for e in zip(p2, p2[1:]):
            used[e][l2] = True
        
        # 12. Сохраняем результат
        result[(s, t)] = {
            'primary': {'wavelength': l1, 'path': p1},
            'backup': {'wavelength': l2, 'path': p2}
        }

    return result

if __name__ == '__main__':
    path = 'test_case/'

    # Чтение файлов для визуализации
    for i in range (1, 10):
        with open(f"test_case/{i}/graph.txt", 'r') as f:
            graph_str = f.read()
        with open(f"test_case/{i}/queries.txt", 'r') as f:
            queries_str = f.read()

        visualize_directed_graph_with_queries_colored_by_wave(graph_str, queries_str)

        solution = solve(path + f'{i}/' + 'graph.txt', path + f'{i}/'+ 'queries.txt')

        print(solution)