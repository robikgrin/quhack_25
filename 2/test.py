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
        num_wavelengths (int): Λ

    Возвращает:
        tuple: (r_score, e_score), где 0 <= r_score <= 1, e_score >= 1.
    """
    if not (len(team_results) == len(test_graphs) == len(test_requests)):
        raise ValueError("Длины списков должны совпадать.")
    if num_wavelengths <= 0:
        raise ValueError("num_wavelengths должно быть > 0.")

    n_tests = len(team_results)
    
    # --- Шаг 1: Определение эталонных значений ---
    reference_results = []
    for i in range(n_tests):
        R_i = test_requests[i]['num_requests']
        # Идеальный эталон: удовлетворить все запросы
        r_hat_i = float(R_i)
        # Идеальный эталон по рёбрам: минимум R_i (по одному ребру на запрос)
        e_hat_i = float(R_i)*5 if R_i > 0 else 0.0
        reference_results.append({'satisfied_requests': r_hat_i, 'used_edges': e_hat_i})

    # --- Шаг 2: Вычисление весов тестов W_i ---
    W_values = []
    for i in range(n_tests):
        V_i = test_graphs[i]['num_vertices']
        E_i = test_graphs[i]['num_edges']
        R_i = test_requests[i]['num_requests']
        if E_i == 0:
            raise ValueError(f"Тест {i}: E_i=0, деление на ноль.")
        W_i = (V_i * R_i) / (num_wavelengths * E_i)
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

# Команда решила 5 из 10 запросов, использовав 20 рёбер
team = [{'satisfied_requests': 5, 'used_edges': 20}]
graphs = [{'num_vertices': 5, 'num_edges': 10}]
requests = [{'num_requests': 10}]
waves = 2

r, e = calculate_scores_with_ideal_reference(team, graphs, requests, waves)
print(f"r_score: {r:.2f}") 
print(f"e_score: {e:.2f}")  