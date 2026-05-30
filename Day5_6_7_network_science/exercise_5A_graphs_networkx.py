"""
Exercise 5A: Graphs as a Data Structure using NetworkX

Topics Covered:
- Nodes and Edges
- Undirected Graph
- Adjacency Matrix
- Adjacency List
- Degree
- Neighbors
- BFS Connectivity
- Paths
- Connected Components
- NetworkX Verification
- Graph Visualization
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# ==========================================================
# (a) BUILD A SMALL SOCIAL NETWORK GRAPH
# ==========================================================

people = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Helen",
    "Ivy",
    "Jack"
]

# 10 Nodes
n = len(people)

# 15 Undirected Edges
edges = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 4),
    (2, 5),
    (3, 4),
    (3, 6),
    (4, 5),
    (4, 7),
    (5, 8),
    (6, 7),
    (6, 9),
    (7, 8),
    (8, 9)
]

# ==========================================================
# ADJACENCY MATRIX (NUMPY)
# ==========================================================

adj_matrix = np.zeros((n, n), dtype=int)

for u, v in edges:
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1

print("\nAdjacency Matrix:")
print(adj_matrix)

# ==========================================================
# ADJACENCY LIST
# ==========================================================

adj_list = {i: [] for i in range(n)}

for u, v in edges:
    adj_list[u].append(v)
    adj_list[v].append(u)

print("\nAdjacency List:")

for node in adj_list:
    print(
        f"{people[node]} -> "
        f"{[people[x] for x in adj_list[node]]}"
    )

# ==========================================================
# NETWORKX GRAPH
# ==========================================================

G = nx.Graph()

for i, person in enumerate(people):
    G.add_node(i, name=person)

G.add_edges_from(edges)

# ==========================================================
# VERIFY ADJACENCY MATRIX == NETWORKX GRAPH
# ==========================================================

nx_matrix = nx.to_numpy_array(G, dtype=int)

print("\nVerification:")

if np.array_equal(adj_matrix, nx_matrix):
    print("✓ NumPy adjacency matrix and NetworkX graph match.")
else:
    print("✗ Representations do not match.")

# ==========================================================
# (b) FUNCTIONS FROM SCRATCH
# ==========================================================

def compute_degree(matrix, node):
    """
    Degree of a node
    """
    return int(np.sum(matrix[node]))


def find_neighbors(matrix, node):
    """
    Return neighbors of node
    """
    nbrs = []

    for j in range(len(matrix)):
        if matrix[node][j] == 1:
            nbrs.append(j)

    return nbrs


def bfs_connected(matrix, start, target):
    """
    Check connectivity using BFS
    """

    visited = set()
    queue = deque()

    visited.add(start)
    queue.append(start)

    while queue:

        current = queue.popleft()

        if current == target:
            return True

        for nbr in find_neighbors(matrix, current):

            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    return False


def bfs_path(matrix, start, target):
    """
    Find path using BFS
    """

    visited = set()
    queue = deque([[start]])

    while queue:

        path = queue.popleft()

        node = path[-1]

        if node == target:
            return path

        if node not in visited:

            visited.add(node)

            for nbr in find_neighbors(matrix, node):

                new_path = list(path)
                new_path.append(nbr)

                queue.append(new_path)

    return None


def connected_components(matrix):
    """
    Find connected components
    """

    n = len(matrix)

    visited = set()

    components = []

    for node in range(n):

        if node not in visited:

            component = []

            queue = deque([node])

            visited.add(node)

            while queue:

                current = queue.popleft()

                component.append(current)

                for nbr in find_neighbors(matrix, current):

                    if nbr not in visited:

                        visited.add(nbr)
                        queue.append(nbr)

            components.append(component)

    return components

# ==========================================================
# TEST CUSTOM FUNCTIONS
# ==========================================================

print("\nNode Degrees:")

for i in range(n):
    print(
        f"{people[i]} : "
        f"{compute_degree(adj_matrix, i)}"
    )

print("\nNeighbors:")

for i in range(n):

    nbrs = find_neighbors(adj_matrix, i)

    print(
        f"{people[i]} : "
        f"{[people[x] for x in nbrs]}"
    )

print("\nBFS Connectivity:")

print(
    "Alice -> Jack:",
    bfs_connected(adj_matrix, 0, 9)
)

print(
    "Bob -> Ivy:",
    bfs_connected(adj_matrix, 1, 8)
)

# ==========================================================
# PATH EXAMPLE
# ==========================================================

path = bfs_path(adj_matrix, 0, 9)

print("\nPath from Alice to Jack:")

print([people[x] for x in path])

# ==========================================================
# CONNECTED COMPONENTS
# ==========================================================

print("\nConnected Components:")

components = connected_components(adj_matrix)

for i, comp in enumerate(components, start=1):

    print(
        f"Component {i}:",
        [people[x] for x in comp]
    )

# ==========================================================
# (c) VERIFY AGAINST NETWORKX
# ==========================================================

print("\nVerification Against NetworkX")

print("\nDegree Check:")

for node in range(n):

    custom_degree = compute_degree(adj_matrix, node)

    nx_degree = G.degree[node]

    print(
        f"{people[node]} "
        f"Custom={custom_degree}, "
        f"NetworkX={nx_degree}"
    )

print("\nNeighbor Check:")

for node in range(n):

    custom_neighbors = sorted(
        find_neighbors(adj_matrix, node)
    )

    nx_neighbors = sorted(
        list(G.neighbors(node))
    )

    print(
        f"{people[node]} : "
        f"{custom_neighbors == nx_neighbors}"
    )

print("\nConnectivity Check:")

pairs = [
    (0, 9),
    (1, 8),
    (2, 7)
]

for s, t in pairs:

    custom = bfs_connected(
        adj_matrix,
        s,
        t
    )

    builtin = nx.has_path(
        G,
        s,
        t
    )

    print(
        f"{people[s]} -> {people[t]} : "
        f"{custom == builtin}"
    )

print("\nConnected Components Check:")

custom_components = len(
    connected_components(adj_matrix)
)

nx_components = nx.number_connected_components(G)

print(
    "Custom Components:",
    custom_components
)

print(
    "NetworkX Components:",
    nx_components
)

# ==========================================================
# (d) VISUALIZE GRAPH
# ==========================================================

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(
    G,
    seed=42
)

labels = {
    i: people[i]
    for i in range(n)
}

nx.draw(
    G,
    pos,
    labels=labels,
    with_labels=True,
    node_size=1800,
    node_color="lightblue",
    edge_color="gray",
    font_size=9,
    width=2
)

plt.title(
    "Social Network Graph\n"
    "(10 Nodes, 15 Edges)"
)

plt.show()
