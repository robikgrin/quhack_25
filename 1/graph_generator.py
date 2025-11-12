import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx


def generate_maxcut_qubo_from_graph(G):
    """
    Преобразует граф G в QUBO-матрицу для задачи MaxCut.
    
    В MaxCut вершины разбиваются на два множества, и целевая функция —
    максимизация веса рёбер между этими множествами.
    
    Формула:
    E(x) = sum_{(i,j) in E} w_{ij} * x_i * (1 - x_j) + x_j * (1 - x_i)
         = sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2*x_i*x_j)
    где x_i ∈ {0, 1} — принадлежность вершины i одному из множеств.

    В QUBO-форме:
    E(x) = x^T * Q * x
    """
    n = G.number_of_nodes()
    Q = np.zeros((n, n))

        # Позиции узлов — можно использовать spring_layout, shell_layout и др.
    pos = nx.spring_layout(G, seed=42)

    # Рисуем узлы (размер зависит от линейного веса)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue')

    # Рисуем рёбра (толщина зависит от |взаимодействия|)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos)

    plt.title("QUBO Network Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    for (i, j) in G.edges():
        Q[i, j] += -1  # коэффициент при x_i * x_j
        Q[j, i] += -1
        Q[i, i] += 1    # коэффициент при x_i
        Q[j, j] += 1     # коэффициент при x_j

    return Q

def generate_random_regular_graph_qubo(n_nodes, degree, seed=None):
    """
    Генерирует QUBO для MaxCut на случайном регулярном графе.
    Все рёбра имеют вес 1.
    """
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    Q = generate_maxcut_qubo_from_graph(G)
    return sp.csr_matrix(Q)

def generate_erdos_renyi_qubo(n_nodes, edge_prob, seed=None):
    """
    Генерирует QUBO для MaxCut на случайном графе Эрдёша–Реньи G(n, p).
    Все рёбра имеют вес 1.
    """
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=seed)
    Q = generate_maxcut_qubo_from_graph(G)
    return sp.csr_matrix(Q)

def generate_grid_qubo(rows, cols, seed=None):
    """
    Генерирует QUBO для MaxCut на двумерной решётке (grid graph).
    Все рёбра имеют вес 1.
    """
    G = nx.grid_2d_graph(rows, cols)
    # Переименовать узлы в числовые индексы
    G = nx.convert_node_labels_to_integers(G)
    Q = generate_maxcut_qubo_from_graph(G)
    return sp.csr_matrix(Q)

def generate_complete_graph_qubo(n_nodes, seed=None):
    """
    Генерирует QUBO для MaxCut на полном графе (K_n).
    Все рёбра имеют вес 1.
    """
    G = nx.complete_graph(n=n_nodes)
    Q = generate_maxcut_qubo_from_graph(G)
    return sp.csr_matrix(Q)

def generate_barabasi_albert_qubo(n_nodes, m_attach, seed=None):
    """
    Генерирует QUBO для MaxCut на графе по модели Барабаши–Альберта.
    Все рёбра имеют вес 1.
    """
    G = nx.barabasi_albert_graph(n=n_nodes, m=m_attach, seed=seed)
    Q = generate_maxcut_qubo_from_graph(G)
    return sp.csr_matrix(Q)

#-> Cycle Graph C8
def cycle_graph_c8():
    G = nx.cycle_graph(8)
    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Cycle Graph C8")
    plt.show()
    return G

# Path Graph P16
def path_graph_p16():
    G = nx.path_graph(16)
    plt.figure(figsize=(12, 2))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=300)
    plt.title("Path Graph P16")
    plt.show()
    return G

#-> Complete Bipartite Graph K8,8
def complete_bipartite_graph_k88():
    G = nx.complete_bipartite_graph(8, 8)
    plt.figure(figsize=(8, 6))
    pos = nx.bipartite_layout(G, nodes=range(8))
    nx.draw(G, pos, with_labels=True, node_color=['lightcoral'] * 8 + ['lightblue'] * 8,
            edge_color='gray', node_size=300)
    plt.title("Complete Bipartite Graph K8,8")
    plt.show()
    return G

#-> Complete Bipartite Graph K8,8
def complete_bipartite_graph_k_nn(n):
    G = nx.complete_bipartite_graph(n, n)
    plt.figure(figsize=(8, 6))
    pos = nx.bipartite_layout(G, nodes=range(n))
    nx.draw(G, pos, with_labels=True, node_color=['lightcoral'] * n + ['lightblue'] * n,
            edge_color='gray', node_size=300)
    plt.title("Complete Bipartite Graph K{},{}".format(n,n))
    plt.show()
    return G

# Star Graph S16
def star_graph_s16():
    G = nx.star_graph(16)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='gold', edge_color='gray', node_size=300)
    plt.title("Star Graph S16")
    plt.show()
    return G

# Grid Graph 8x4
def grid_graph_8x4():
    G = nx.grid_graph(dim=[8, 4])
    plt.figure(figsize=(12, 6))
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    plt.title("Grid Graph 8x4")
    plt.show()
    return G

# Grid Graph 8x4
def grid_graph_nxm(n,m):
    G = nx.grid_graph(dim=[n, m])
    plt.figure(figsize=(12, 6))
    pos = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    plt.title("Grid Graph {}x{}".format(n,m))
    plt.show()
    return G


#-> 4-Regular Graph with 8 Vertices
def regular_graph_4_8():
    G = nx.random_regular_graph(d=4, n=8, seed=42)
    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500)
    plt.title("4-Regular Graph with 8 Vertices")
    plt.show()
    return G

#-> Cubic (3-Regular) Graph with 16 Vertices
def cubic_graph_3_16():
    G = nx.random_regular_graph(d=3, n=16, seed=42)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=300)
    plt.title("Cubic (3-Regular) Graph with 16 Vertices")
    plt.show()
    return G

# Disjoint Union of Four C4 Cycles
def disjoint_union_c4():
    cycles = [nx.cycle_graph(4) for _ in range(4)]
    G = nx.disjoint_union_all(cycles)
    plt.figure(figsize=(12, 6))
    pos = {}
    shift_x = 0
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        pos_sub = nx.circular_layout(subgraph, scale=1, center=(shift_x, 0))
        pos.update(pos_sub)
        shift_x += 3
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    plt.title("Disjoint Union of Four C4 Cycles")
    plt.show()
    return G

# Complete Bipartite Graph K16,16
def complete_bipartite_graph_k1616():
    G = nx.complete_bipartite_graph(16, 16)
    plt.figure(figsize=(12, 6))
    pos = nx.bipartite_layout(G, nodes=range(16))
    nx.draw(G, pos, with_labels=False, node_color=['lightcoral'] * 16 + ['lightblue'] * 16,
            edge_color='gray', node_size=100)
    plt.title("Complete Bipartite Graph K16,16")
    plt.show()
    return G

# 5-Dimensional Hypercube Graph Q5
def hypercube_graph_q5():
    G = nx.hypercube_graph(5)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=200)
    plt.title("5-Dimensional Hypercube Graph Q5")
    plt.show()
    return G

# Tree Graph with 8 Vertices
def tree_graph_8():
    G = nx.balanced_tree(r=2, h=2)
    G.add_edge(6, 7)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    plt.title("Tree Graph with 8 Vertices")
    plt.show()
    return G

# Wheel Graph W16
def wheel_graph_w16():
    G = nx.wheel_graph(16)
    plt.figure(figsize=(8, 8))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=300)
    plt.title("Wheel Graph W16")
    plt.show()
    return G