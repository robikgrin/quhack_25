from solution_rwa import *
import time

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


if __name__ == '__main__':
    team_results = []
    graphs = []
    requests = []
    waves = []
    # Чтение файлов для визуализации
    for i in range (1, 51):
        path = f'test_case/{i}'
        with open(f"{path}/graph.txt", 'r') as f:
            graph_str = f.read()
            v_num, r_num = list(map(int, graph_str.split()[:2]))

        with open(f"{path}/queries.txt", 'r') as f:
            queries_str = f.read()
            waves_num, req_num = list(map(int, queries_str.split()[:2]))

        graphs.append({'num_vertices': v_num, 'num_edges': r_num})
        requests.append({'num_requests': req_num})
        waves.append(waves_num)

        
        start = time.time()
        solution = solve(path + '/graph.txt', path + '/queries.txt')
        final = time.time()

        print('Time calc:', final - start)
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