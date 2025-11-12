import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import heapq

from solution_sim import solve_qubo

def Dijkstra(graph, source, target=None):
    n = graph.shape[0]
    distances = np.full(n, np.inf)
    predecessors = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    
    distances[source] = 0
    heap = [(0, source)]
    
    while heap:
        current_dist, current_node = heapq.heappop(heap)
        
        if target is not None and current_node == target:
            break
            
        if visited[current_node]:
            continue
        visited[current_node] = True
        
        start = graph.indptr[current_node]
        end = graph.indptr[current_node + 1]
        
        for i in range(start, end):
            neighbor = graph.indices[i]
            weight = graph.data[i]
            if weight == 0:
                continue
            if not visited[neighbor]:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))
    
    return distances, predecessors

def reconstruct_path(predecessors, source, target):
    if predecessors[target] == -1 and source != target:
        return None 
    path = []
    current = target
    while current != -1:
        path.append(current)
        current = predecessors[current]
    return path[::-1]

def dijkstra_shortest_path(graph, source, target):
    distances, predecessors = Dijkstra(graph, source, target)
    if distances[target] == np.inf:
        return None, np.inf
    path = reconstruct_path(predecessors, source, target)
    return path, distances[target]

def Yen_k_shortest_paths(graph, source, target, K=3):
    path0, _ = dijkstra_shortest_path(graph, source, target)
    if path0 is None:
        return []
    
    A = [path0]
    B = []

    for k in range(1, K):
        for i in range(len(A[k - 1]) - 1):
            spur_node = A[k - 1][i]
            root_path = A[k - 1][:i + 1]

            removed_edges = []
            for path in A:
                if len(path) > i and path[:i + 1] == root_path:
                    u, v = path[i], path[i + 1]
                    if graph[u, v] != 0:
                        removed_edges.append((u, v, graph[u, v]))
                        graph[u, v] = 0

            spur_path, _ = dijkstra_shortest_path(graph, spur_node, target)

            for u, v, w in removed_edges:
                graph[u, v] = w

            if spur_path is not None:
                total_path = root_path + spur_path[1:]

                if total_path not in A and not any(p == total_path for _, p in B):
                    total_len = sum(graph[u, v] for u, v in zip(total_path[:-1], total_path[1:]))
                    heapq.heappush(B, (total_len, total_path))

        if not B:
            break

        _, best_path = heapq.heappop(B)
        A.append(best_path)

    return A

def paths_share_edge(p1, p2):
    return bool(set(zip(p1, p1[1:])) & set(zip(p2, p2[1:])))

def available(path, wl, used):
    for e in zip(path, path[1:]):
        if used[e][wl]:
            return False
    return True

def read_graph_from_file(filename):
    with open(filename, 'r') as f:
        n_vertices, n_edges = map(int, f.readline().strip().split())
        edges = []
        for line in f:
            if line.strip():
                u, v = map(int, line.strip().split())
                edges.append((u, v))
    return n_vertices, edges

def create_graph(n_vertices, edges):
    row, col = zip(*edges) if edges else ([], [])
    data = [1] * len(edges)
    graph = sp.csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices), dtype=int)
    return graph


def solve(graph_file: str, requests_file: str):
    n_vertices, edges = read_graph_from_file(graph_file)
    G = create_graph(n_vertices, edges)  

    with open(requests_file) as f:
        lines = f.read().strip().splitlines()
    Lambda = int(lines[0])
    R = int(lines[1])
    requests = [tuple(map(int, lines[2 + i].split())) for i in range(R)]

    options = {}
    max_path_length = 0

    for s, t in requests:
        try:
            all_paths = Yen_k_shortest_paths(G.copy(), s, t, K=10)
        except Exception:
            all_paths = []

        pairs = []
        count = 0
        for i in range(len(all_paths)):
            for j in range(i + 1, len(all_paths)):
                p1, p2 = all_paths[i], all_paths[j]
                if not paths_share_edge(p1, p2):
                    pairs.append((p1, p2))
                    max_path_length = max(max_path_length, len(p1) + len(p2))
                    count += 1
                    if count >= 3:
                        break
            if count >= 3:
                break

        options[(s, t)] = pairs

    requests_sorted = sorted(requests, key=lambda r: len(options[r]))

    used = defaultdict(lambda: [False] * Lambda)

    M = max_path_length if max_path_length > 0 else 1
    alpha = 1.0
    beta = R * (M - 2) + 3
    rho = beta * (R + 1) + 100

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

        n = len(candidates)
        row_indices = []
        col_indices = []
        data = []

        for i in range(n):
            p1, p2, _, _ = candidates[i]
            cost = alpha * ((len(p1) - 1) + (len(p2) - 1)) - beta
            row_indices.append(i)
            col_indices.append(i)
            data.append(cost)

        for i in range(n):
            for j in range(i + 1, n):
                p1_i, p2_i, l1_i, l2_i = candidates[i]
                p1_j, p2_j, l1_j, l2_j = candidates[j]

                conflict = False
                if l1_i == l1_j and paths_share_edge(p1_i, p1_j):
                    conflict = True
                if l2_i == l2_j and paths_share_edge(p2_i, p2_j):
                    conflict = True
                if l1_i == l2_j and paths_share_edge(p1_i, p2_j):
                    conflict = True
                if l2_i == l1_j and paths_share_edge(p2_i, p1_j):
                    conflict = True

                if conflict:
                    row_indices.extend([i, j])
                    col_indices.extend([j, i])
                    data.extend([rho, rho])

        Q = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        solution = solve_qubo(Q)

        chosen = None
        for i in range(len(solution)):
            if solution[i] == 1:
                chosen = i
                break

        if chosen is None:
            costs = [len(c[0]) + len(c[1]) for c in candidates]
            chosen = int(np.argmin(costs))

        if chosen is None:
            continue

        p1, p2, l1, l2 = candidates[chosen]
        for e in zip(p1, p1[1:]):
            used[e][l1] = True
        for e in zip(p2, p2[1:]):
            used[e][l2] = True

        result[(s, t)] = {
            'primary': {'wavelength': l1, 'path': p1},
            'backup': {'wavelength': l2, 'path': p2}
        }

    return result