import numpy as np

def calculate_scores(team_results, test_graphs, test_requests, num_wavelengths):
    """
    Вычисляет итоговые метрики r_score и e_score для команды по всем тестам.
    Референсные значения (r̂_i, ê_i) оцениваются с помощью простой эвристической модели,
    основанной на структуре графа и количестве волн.

    Аргументы:
        team_results (list of dict): Результаты команды для каждого теста.
            Каждый словарь содержит:
            - 'satisfied_requests' (int): Количество удовлетворённых запросов (r_i).
            - 'used_edges' (int): Количество использованных рёбер (e_i).
        test_graphs (list of dict): Информация о графах для каждого теста.
            Каждый словарь содержит:
            - 'num_vertices' (int): Количество вершин |V_i|.
            - 'num_edges' (int): Количество рёбер |E_i|.
        test_requests (list of dict): Информация о запросах для каждого теста.
            Каждый словарь содержит:
            - 'num_requests' (int): Количество запросов |R_i|.
        num_wavelengths (int): Количество волн Λ (одинаковое для всех тестов).

    Возвращает:
        tuple: (r_score, e_score) - итоговые метрики команды.
    """
    if not (len(team_results) == len(test_graphs) == len(test_requests)):
        raise ValueError("Длины списков результатов и тестов должны совпадать.")

    if num_wavelengths <= 0:
        raise ValueError("Количество волн (num_wavelengths) должно быть положительным.")

    # Генерация референсных значений для каждого теста
    reference_results = []
    for i in range(len(team_results)):
        V_i = test_graphs[i]['num_vertices']
        E_i = test_graphs[i]['num_edges']
        R_i = test_requests[i]['num_requests']

        # Оценка референсного количества удовлетворённых запросов (r̂_i)
        # Эвристика: идеальный алгоритм может удовлетворить не все запросы из-за ограничения волн.
        # Максимальное количество запросов, которое можно удовлетворить, ограничено:
        # 1. Числом запросов R_i.
        # 2. Общим "ресурсом" волн: каждая волна может обслужить один путь.
        # 3. Сложностью путей: средняя длина пути в графе примерно равна log(V_i) или sqrt(V_i).
        # Используем простую модель: r̂_i = min(R_i, num_wavelengths * (E_i / V_i))
        # Это предполагает, что среднее количество рёбер на путь равно V_i / E_i? Нет, это неверно.
        # Лучше: r̂_i = min(R_i, num_wavelengths * (E_i / avg_path_length))
        # Но avg_path_length неизвестна. Используем грубую оценку: avg_path_length ~ V_i / 2.
        # Тогда r̂_i = min(R_i, num_wavelengths * E_i / (V_i / 2)) = min(R_i, 2 * num_wavelengths * E_i / V_i)
        # Это слишком завышает. Более консервативная оценка: r̂_i = min(R_i, num_wavelengths * sqrt(E_i))
        # Или: r̂_i = min(R_i, num_wavelengths * V_i // 2) — если граф плотный.
        # Применим следующую эвристику:
        # Если граф разреженный (E_i < V_i), то r̂_i = min(R_i, num_wavelengths * E_i)
        # Если граф плотный (E_i >= V_i), то r̂_i = min(R_i, num_wavelengths * V_i)
        # Это учитывает, что в разреженном графе каждый путь использует много рёбер, а в плотном — мало.
        # Другой подход: r̂_i = min(R_i, num_wavelengths * max(1, E_i // V_i))
        # Попробуем более сложную эвристику:
        # r̂_i = min(R_i, num_wavelengths * (E_i / max(1, V_i - 1)) * 0.8)
        # Умножаем на 0.8, чтобы учесть, что не все рёбра могут быть использованы эффективно.
        # Или: r̂_i = min(R_i, num_wavelengths * (E_i / V_i) * 2) — но это может быть больше R_i.
        # Самый простой и реалистичный вариант: использовать минимальную границу, основанную на том,
        # что для одного запроса нужно минимум одно ребро, а всего доступно num_wavelengths * E_i "слотов".
        # Тогда r̂_i = min(R_i, num_wavelengths * E_i)
        # Но это слишком оптимистично. Уменьшим коэффициент.
        # Используем: r̂_i = min(R_i, num_wavelengths * E_i * 0.5)
        # Это предполагает, что в среднем на один путь используется 2 ребра.
        # Для большей точности, можно использовать: r̂_i = min(R_i, num_wavelengths * E_i / 2)
        # Но это может быть меньше 1. Поэтому: r̂_i = max(1, min(R_i, num_wavelengths * E_i // 2))
        # Однако, если R_i мало, то r̂_i = R_i.
        # Итоговая эвристика:
        if E_i == 0:
            r_hat_i = 0.0
        else:
            # Оценка максимального числа запросов, которые можно удовлетворить
            # Учитываем, что каждый путь использует в среднем несколько рёбер.
            # Коэффициент 0.7 — эмпирический, можно подбирать.
            estimated_max_requests = num_wavelengths * E_i * 0.7
            r_hat_i = min(float(R_i), estimated_max_requests)

        # Оценка референсного количества использованных рёбер (ê_i)
        # Эвристика: минимальное количество рёбер, необходимое для удовлетворения r̂_i запросов.
        # Предполагаем, что средняя длина пути равна 2 (для простоты).
        # Тогда ê_i = r̂_i * 2
        # Но если r̂_i = 0, то ê_i = 0.
        if r_hat_i == 0:
            e_hat_i = 0.0
        else:
            # Средняя длина пути: можно оценить как log(V_i) или sqrt(V_i), но проще взять 2.
            avg_path_length = 2.0
            # Умножаем на коэффициент, чтобы учесть, что пути могут пересекаться и использовать общие рёбра.
            # Коэффициент 0.8 — эмпирический.
            e_hat_i = r_hat_i * avg_path_length * 0.8

        reference_results.append({'satisfied_requests': r_hat_i, 'used_edges': e_hat_i})

    # 1. Вычисление внутренних весов (сложности) для каждого теста W_i
    W_values = []
    for i in range(len(team_results)):
        V_i = test_graphs[i]['num_vertices']
        E_i = test_graphs[i]['num_edges']
        R_i = test_requests[i]['num_requests']

        if E_i == 0:
            raise ValueError(f"Количество рёбер в тесте {i} равно 0, деление на ноль невозможно.")

        W_i = (V_i * R_i) / (num_wavelengths * E_i)
        W_values.append(W_i)

    W_min = min(W_values)
    W_max = max(W_values)

    epsilon = 1e-9
    w_values = []

    if W_max == W_min:
        # Все тесты одинаковой сложности
        w_values = [1.0 for _ in W_values]
    else:
        for W_i in W_values:
            w_i = (W_i - W_min) / (W_max - W_min) + epsilon
            w_values.append(w_i)

    W = sum(w_values)
    if W == 0:
        raise ValueError("Сумма нормализованных весов равна 0.")

    # 2. Вычисление r_score и e_score
    numerator_r = 0.0
    numerator_e = 0.0

    for i in range(len(team_results)):
        r_i = team_results[i]['satisfied_requests']
        e_i = team_results[i]['used_edges']
        r_hat_i = reference_results[i]['satisfied_requests']
        e_hat_i = reference_results[i]['used_edges']
        w_i = w_values[i]

        # Обработка случая, когда количество запросов R_i = 0
        if test_requests[i]['num_requests'] == 0:
            # Если нет запросов, то удовлетворить можно только 0, и использовать рёбер не нужно.
            if r_i != 0:
                ratio_r = 0.0  # Нельзя удовлетворить запросы, которых нет.
            else:
                ratio_r = 1.0  # Идеально: удовлетворено 0 запросов из 0.

            if e_i == 0:
                ratio_e = 1.0  # Идеально: ничего не использовано.
            else:
                ratio_e = float('inf')  # Любое использование рёбер — плохо.
        else:
            # Обычный случай
            if r_hat_i == 0:
                ratio_r = 0.0
            else:
                ratio_r = r_i / r_hat_i

            if e_hat_i == 0:
                if e_i == 0:
                    ratio_e = 1.0
                else:
                    ratio_e = float('inf')
            else:
                ratio_e = e_i / e_hat_i

        numerator_r += w_i * ratio_r
        numerator_e += w_i * ratio_e

    r_score = numerator_r / W
    e_score = numerator_e / W

    return r_score, e_score

# Пример использования:
if __name__ == "__main__":
    # Пример данных для 2 тестов
    team_results = [
        {'satisfied_requests': 5, 'used_edges': 10},
        {'satisfied_requests': 8, 'used_edges': 15}
    ]
    test_graphs = [
        {'num_vertices': 5, 'num_edges': 8},
        {'num_vertices': 6, 'num_edges': 10}
    ]
    test_requests = [
        {'num_requests': 2},  # Команда удовлетворила 5 из 6
        {'num_requests': 3}   # Команда удовлетворила 8 из 8
    ]
    num_wavelengths = 2

    r_score, e_score = calculate_scores(
        team_results,
        test_graphs,
        test_requests,
        num_wavelengths
    )

    print(f"r_score: {r_score:.6f}")  # Чем ближе к 1, тем лучше (по запросам)
    print(f"e_score: {e_score:.6f}")  # Чем ближе к 0, тем лучше (по рёбрам)