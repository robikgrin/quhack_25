from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# Маппинг имён в индексы
nodes = ['O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
n = len(nodes)

# Матрица расстояний (из твоей таблицы)
# Используем большое число вместо "-"
INF = 10**9
dist_matrix = [
    [0,  15, 30, 25, 40, 17, 32, INF, 28, INF, 26, 31, 42, INF, 35],  # O
    [15, 0,  8,  INF,18, INF,INF,INF, 20, INF, 11, 19, INF,26, INF],  # A
    [30, 8,  0,  9,  16, 14, INF,INF, 14, INF, INF,INF,23, INF,INF],  # B
    [25, INF,9,  0,  INF,21, INF,16, INF,30, INF,33, INF,40, 19],    # C
    [40, 18, 16, INF,0,  INF,22, INF,INF,INF, INF,29, 25, 38, INF],  # D
    [17, INF,14, 21, INF,0,  INF,INF, 27,14, INF,27, INF,INF,24],    # E
    [32, INF,INF,INF,22, INF,0,  10, INF,INF,22, 26, INF,35, 31],    # F
    [INF,INF,INF,16, INF,INF,10, 0,  24,17, INF,INF,29, 38, INF],    # G
    [28, 20, 14, INF,INF,27, INF,24, 0,  INF,19, INF,INF,29, 23],    # H
    [INF,INF,INF,30, INF,14, INF,17, INF,0,  24, 30, INF,39, INF],   # I
    [26, 11, INF,INF,INF,INF,22, INF,19,24, 0,  INF,INF,33, INF],    # J
    [31, 19, INF,33, 29, 27, 26, INF,INF,30, INF,0,  13, INF,20],    # K
    [42, INF,23, INF,25, INF,INF,29, INF,INF,INF,13, 0,  24, INF],   # L
    [INF,26, INF,40, 38, INF,35, 38, 29,39, 33, INF,24, 0,  17],     # M
    [35, INF,INF,19, INF,24, 31, INF,23,INF, INF,20, INF,17, 0]      # N
]

# Преобразуем в numpy и заменим INF на большое число (OR-Tools не любит inf)
dist_np = np.array(dist_matrix, dtype=np.int64)
dist_np[dist_np == INF] = 999999  # большое число

# Применяем Флойда–Уоршелла для восстановления кратчайших путей
def floyd_warshall(dist):
    n = len(dist)
    dist = dist.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

full_dist = floyd_warshall(dist_np)

# Проверим, нет ли недостижимых узлов
if np.any(full_dist >= 999999):
    print("⚠️  В графе есть недостижимые узлы! TSP невозможен.")
    exit()

# --- Решаем TSP с помощью OR-Tools ---
def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 маршрут, старт с 0 (O)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Параметры поиска
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        print("❌ Решение не найдено")
        return None, None

    index = routing.Start(0)
    route = []
    total_distance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    route.append(manager.IndexToNode(index))  # возврат в старт (опционально)
    return route[:-1], total_distance  # убираем дублирование старта в конце

route, length = solve_tsp(full_dist)

if route:
    route_names = [nodes[i] for i in route]
    print("✅ Найден маршрут:")
    print(" → ".join(route_names))
    print(f"Длина маршрута: {length}")
    # =============================
    # ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ OR-Tools
    # =============================

    import networkx as nx
    import matplotlib.pyplot as plt

    # Создаём граф из ИСХОДНОЙ матрицы (только существующие рёбра)
    G = nx.Graph()

    # Добавляем узлы
    for name in nodes:
        G.add_node(name)

    # Добавляем рёбра (игнорируем INF / 999999)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] != INF:
                G.add_edge(nodes[i], nodes[j], weight=dist_matrix[i][j])

    # Генерируем позиции узлов (фиксированный seed для повторяемости)
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Нарисуем всё
    plt.figure(figsize=(12, 10))

    # 1. Все существующие рёбра — тонкие серые линии
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.0, alpha=0.7)

    # 2. Узлы
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=900, edgecolors='black', linewidths=1.2)

    # 3. Метки узлов
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # 4. Маршрут от OR-Tools — красные стрелки
    if route:
        # Создаём ориентированный путь: [v0, v1, ..., v14] → рёбра (v0→v1), (v1→v2), ..., (v14→v0)
        route_edges = [(nodes[route[i]], nodes[route[(i + 1) % len(route)]]) for i in range(len(route))]
        
        # Чтобы стрелки не накладывались на рёбра, используем DiGraph для отрисовки
        G_route = nx.DiGraph()
        G_route.add_edges_from(route_edges)
        
        # Рисуем маршрут
        nx.draw_networkx_edges(
            G_route, pos,
            edge_color='red',
            width=3.0,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',  # слегка изогнём стрелки
            alpha=0.9
        )

    # Настройки графика
    plt.title(f"Маршрут коммивояжёра (OR-Tools)\nДлина: {length}", fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Сохраняем и показываем
    plt.savefig("tsp_ortools_solution.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("❌ Не удалось найти маршрут")
