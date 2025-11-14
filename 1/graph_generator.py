import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx

def generate_maxcut_qubo_from_graph(G, position = None, vis = False):
    """
    Преобразует граф G в QUBO-матрицу для задачи MaxCut.
    
    Формула для функции оценки

        E(x) = sum_{(i,j) in E} w_{ij} * x_i * (1 - x_j) + x_j * (1 - x_i)
            = sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2*x_i*x_j)
        где x_i ∈ {0, 1} — принадлежность вершины i одному из множеств.

    В QUBO-форме:

        E(x) = x^T * Q * x
    """
    n = G.number_of_nodes()
    Q = np.zeros((n, n))

    if vis:
        if position:
            pos = position
        else:
            pos = nx.spring_layout(G, seed=42)  

        plt.figure(figsize=(8,6), dpi = 400)

        nx.draw_networkx_nodes(G, pos, node_color='lightgreen')

        nx.draw_networkx_edges(G, pos, edge_color='gray')
        nx.draw_networkx_labels(G, pos)

        plt.title("QUBO Network Visualization", fontsize = 18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    for (i, j) in G.edges():
        Q[i, j] += -1  # коэффициент при x_i * x_j
        Q[j, i] += -1
        Q[i, i] += 1    # коэффициент при x_i
        Q[j, j] += 1     # коэффициент при x_j
    return Q

def generate_random_regular_graph_qubo(n_nodes, degree, seed=None, vis=False):
    """
    Генерирует QUBO для MaxCut на случайном регулярном графе.
    """
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    Q = generate_maxcut_qubo_from_graph(G, vis)
    return sp.csr_matrix(Q), G

def generate_erdos_renyi_qubo(n_nodes, edge_prob, seed=None, vis=False):
    """
    Генерирует QUBO для MaxCut на случайном графе Эрдёша–Реньи G(n, p).
    """
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=seed)
    Q = generate_maxcut_qubo_from_graph(G, vis)
    return sp.csr_matrix(Q), G

def generate_grid_qubo(rows, cols, seed=None, vis = False):
    """
    Генерирует QUBO для MaxCut на двумерной решётке (grid graph).
    """
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G)
    pos = {}
    temp = 0
    for row in range(rows):
        for col in range(cols):
            pos[temp] = (col, -row)
            temp += 1
    Q = generate_maxcut_qubo_from_graph(G, pos, vis)
    return sp.csr_matrix(Q), G

def generate_complete_graph_qubo(n_nodes, seed=None, vis = False):
    """
    Генерирует QUBO для MaxCut на полном графе (K_n).
    """
    G = nx.complete_graph(n=n_nodes)
    Q = generate_maxcut_qubo_from_graph(G, vis)
    return sp.csr_matrix(Q), G

def complete_bipartite_graph_k_nn(n, seed = None, vis = False):
    G = nx.complete_bipartite_graph(n, n)
    pos = nx.bipartite_layout(G, nodes=range(n))

    Q = generate_maxcut_qubo_from_graph(G, pos, vis)
    return sp.csr_matrix(Q), G