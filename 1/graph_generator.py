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