import random
import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_directed_graph_and_queries_files(n_vertices, n_edges, n_waves=1, seed=None, output_dir="generated_data"):
    """
    Генерирует два файла в требуемом формате:
    - graph.txt: Описание графа.
    - queries.txt: Данные о волнах и запросах.
    
    Аргументы:
        n_vertices (int): Количество вершин (нумерация с 0).
        n_edges (int): Количество рёбер.
        n_waves (int): Количество "волн" (запросов на разных волнах).
        seed (int, optional): Сид для воспроизводимости. По умолчанию None.
        output_dir (str): Директория для сохранения файлов.
    """
    if seed is not None:
        random.seed(seed)
    
    # Проверки
    if n_vertices < 1:
        raise ValueError("Количество вершин должно быть >= 1")
    if n_edges < 0:
        raise ValueError("Количество рёбер не может быть отрицательным")
    if n_waves < 0:
        raise ValueError("Количество волн не может быть отрицательным")
    
    max_possible_edges = n_vertices * (n_vertices - 1)
    if n_edges > max_possible_edges:
        raise ValueError(f"Максимальное количество рёбер для {n_vertices} вершин: {max_possible_edges}")
    
    # Создание директории
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерация рёбер (без петель)
    edges = set()
    while len(edges) < n_edges:
        u = random.randint(0, n_vertices - 1)
        v = random.randint(0, n_vertices - 1)
        if u != v:
            edges.add((u, v))
    
    # Генерация запросов (между разными вершинами)
    queries = set()
    while len(queries) < n_waves:
        u = random.randint(0, n_vertices - 1)
        v = random.randint(0, n_vertices - 1)
        if u != v:  # Запрещаем запросы типа A->A
            queries.add((u, v))
    
    # Запись в файл graph.txt
    graph_path = os.path.join(output_dir, "graph.txt")
    with open(graph_path, 'w') as f:
        f.write(f"{n_vertices} {n_edges}\n")
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
    
    # Запись в файл queries.txt
    queries_path = os.path.join(output_dir, "queries.txt")
    with open(queries_path, 'w') as f:
        f.write(f"{n_waves}\n")  # Число волн
        f.write(f"{len(queries)}\n")  # Число запросов
        for u, v in sorted(queries):
            f.write(f"{u} {v}\n")
    
    print(f"Файлы успешно созданы в '{output_dir}':")
    print(f"  - graph.txt")
    print(f"  - queries.txt")

def visualize_directed_graph_with_queries_colored_by_wave(graph_str, queries_str, title="Направленный граф с запросами"):
    """
    Визуализирует направленный граф из двух строк, выделяя точки запросов разными цветами по волнам.
    Одинаковые запросы (одинаковые пары вершин) рисуются одним цветом.
    Запросы только между разными вершинами.
    
    Аргументы:
        graph_str (str): Строка с описанием графа.
        queries_str (str): Строка с дополнительными данными (волнами и запросами).
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
            raise ValueError(f"Петля обнаружена: {u}->{v}. Запросы должны быть между разными вершинами.")
        if u < 0 or u >= n_vertices or v < 0 or v >= n_vertices:
            raise ValueError(f"Вершина {u} или {v} выходит за пределы диапазона [0, {n_vertices-1}]")
        edges.append((u, v))
    
    G.add_edges_from(edges)
    
    # Парсинг queries_str
    query_lines = queries_str.strip().split('\n')
    if len(query_lines) < 3:
        raise ValueError("queries_str содержит недостаточно строк")
    
    try:
        n_waves = int(query_lines[0])
        n_queries = int(query_lines[1])
    except (ValueError, IndexError):
        raise ValueError("Первые две строки queries_str должны содержать целые числа: количество волн и запросов")
    
    if len(query_lines) < 2 + n_queries:
        raise ValueError(f"Ожидается {n_queries} запросов в queries_str, но строк недостаточно")
    
    queries = []
    for i in range(2, 2 + n_queries):
        parts = query_lines[i].split()
        if len(parts) != 2:
            raise ValueError(f"Строка запроса {i+1} в queries_str должна содержать две вершины: 'источник цель'")
        u, v = int(parts[0]), int(parts[1])
        if u == v:
            raise ValueError(f"Запрос с петлёй обнаружен: {u}->{v}. Запросы должны быть между разными вершинами.")
        if u < 0 or u >= n_vertices or v < 0 or v >= n_vertices:
            raise ValueError(f"Вершина {u} или {v} выходит за пределы диапазона [0, {n_vertices-1}]")
        queries.append((u, v))
    
    # Настройка визуализации
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Рисуем основные элементы графа
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray', edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='darkgray', width=2)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
    
    # Выделение точек запросов по уникальным парам (одинаковые запросы — один цвет)
    unique_queries = list(set(queries))
    color_map = {}
    colors = plt.cm.Set1(range(len(unique_queries)))
    
    for idx, query in enumerate(unique_queries):
        color_map[query] = colors[idx]
    
    # Для каждого запроса выделяем его источники и цели одним цветом
    for wave_idx, (start, end) in enumerate(queries):
        query_pair = (start, end)
        color = color_map[query_pair]
        
        nx.draw_networkx_nodes(G, pos, nodelist=[start], node_size=800, node_color=color, edgecolors='black', linewidths=3, label=f'Запрос {wave_idx+1}: {start}->{end}')
        nx.draw_networkx_nodes(G, pos, nodelist=[end], node_size=800, node_color=color, edgecolors='black', linewidths=3)
    
    # Добавляем легенду
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, loc='upper right', fontsize=12)
    
    plt.title(title, fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Генерация файлов
    for i in range(1, 10):
        generate_directed_graph_and_queries_files(
            n_vertices=10,
            n_edges=30,
            n_waves=4,
            output_dir=f"test_case/{i}/"
        )
    
    
    # Чтение файлов для визуализации
    with open("test_case/graph.txt", 'r') as f:
        graph_str = f.read()
    with open("test_case/queries.txt", 'r') as f:
        queries_str = f.read()
    
    # Визуализация
    visualize_directed_graph_with_queries_colored_by_wave(
        graph_str,
        queries_str,
        "Граф с запросами между разными вершинами"
    )