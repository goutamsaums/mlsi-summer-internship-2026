# Day 5, 6, 7 – Network Science & Graph Theory

## Overview

Days 5–7 focused on the foundations of **Network Science** and **Graph Theory** using Python.

The objective was to understand how networks are represented, analyzed, generated, and visualized using NumPy, NetworkX, and Matplotlib.

Topics covered:

* Graphs as Data Structures
* Nodes and Edges
* Directed vs Undirected Graphs
* Adjacency Matrix
* Adjacency List
* Degree and Neighbors
* Paths and Connectivity
* Connected Components
* Breadth-First Search (BFS)
* NetworkX Library
* Kronecker Graphs
* Graph Visualization

---

# Learning Resources

### Graph Theory

1. William Fiset Graph Theory Playlist (First 5 Lectures)
2. Graph Theory Basics
3. Graph Data Structure Implementations

### NetworkX

* NetworkX Tutorial Playlist
* NetworkX Official Documentation

---

# 1. Graph Fundamentals

A graph consists of:

* **Nodes (Vertices)** → entities
* **Edges (Links)** → relationships between entities

Examples:

* Social Networks
* Communication Networks
* Transportation Networks
* Biological Networks

---

## Graph Types

### Undirected Graph

Connections have no direction.

Example:

Alice ↔ Bob

Friendship is mutual.

### Directed Graph

Connections have direction.

Example:

Alice → Bob

Twitter follows are directed.

---

# 2. Graph Representations

## Adjacency Matrix

A square matrix of size n × n.

A[i][j] = 1 if an edge exists between nodes i and j.

Properties:

* Space Complexity: O(n²)
* Edge Lookup: O(1)
* Suitable for dense graphs

Example:

```python
adj_matrix = np.zeros((n, n), dtype=int)

for u, v in edges:
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1
```

---

## Adjacency List

Stores neighbors of each node.

Example:

```python
adj_list = {i: [] for i in range(n)}

for u, v in edges:
    adj_list[u].append(v)
    adj_list[v].append(u)
```

Properties:

* Space Complexity: O(V + E)
* Efficient for sparse graphs

---

# 3. Graph Properties

## Degree

Number of edges connected to a node.

Example:

```python
def compute_degree(matrix, node):
    degree = 0
    for value in matrix[node]:
        degree += value
    return degree
```

---

## Neighbors

Nodes directly connected to a node.

Example:

```python
def find_neighbors(matrix, node):
    neighbors = []

    for j in range(len(matrix)):
        if matrix[node][j] == 1:
            neighbors.append(j)

    return neighbors
```

---

## Paths

A sequence of nodes connecting two nodes.

Example:

A → B → C → D

Path length = 3 edges

---

## Connected Components

A connected component is a set of nodes where every node can reach every other node.

If all nodes belong to one component, the graph is connected.

---

# 4. Breadth-First Search (BFS)

BFS explores nodes level by level.

Applications:

* Connectivity checking
* Shortest path in unweighted graphs
* Connected component detection

Complexity:

* Time: O(V + E)
* Space: O(V)

Example:

```python
def bfs_connected(matrix, start, target):

    if start == target:
        return True

    visited = set([start])
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for neighbor in find_neighbors(matrix, current):

            if neighbor == target:
                return True

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False
```

---

# 5. NetworkX Basics

NetworkX is a Python library for graph creation and analysis.

## Graph Creation

```python
import networkx as nx

G = nx.Graph()

G.add_nodes_from([0,1,2])
G.add_edges_from([(0,1),(1,2)])
```

## Useful Functions

```python
G.degree[node]

G.neighbors(node)

nx.has_path(G, source, target)

nx.shortest_path(G, source, target)

nx.connected_components(G)

nx.density(G)

nx.average_clustering(G)
```

---

# Exercise 5A – Graph Data Structures & NetworkX

## Objective

Create and analyze a small social network graph.

### Tasks Completed

### Graph Construction

Created:

* 10 nodes
* 15 friendship edges

Represented as:

* Adjacency Matrix (NumPy)
* Adjacency List
* NetworkX Graph

Verified both representations encode identical information.

### Functions Implemented From Scratch

Implemented:

* compute_degree()
* find_neighbors()
* bfs_connected()
* bfs_find_path()
* find_connected_components()

### Verification Using NetworkX

Compared custom implementations with:

* G.degree
* G.neighbors()
* nx.has_path()
* nx.shortest_path()
* nx.connected_components()

Result:

All outputs matched NetworkX.

### Visualization

Generated:

* Spring Layout
* Circular Layout
* Degree Distribution Histogram

---

# Kronecker Graphs

## Concept

Kronecker Graphs are generated using the Kronecker Product of a small initiator matrix.

They exhibit:

* Self-similarity
* Hierarchical structure
* Broad degree distributions
* Realistic network characteristics

---

## Initiator Matrix

```python
K1 = np.array([
    [1,1],
    [1,0]
])
```

---

## Recursive Expansion

```python
K2 = np.kron(K1, K1)

K3 = np.kron(K2, K1)
```

This creates increasingly larger adjacency matrices.

---

# Exercise 5B – Kronecker Graph Generation

## Objective

Generate and analyze Kronecker graphs.

### Tasks Completed

### Graph Generation

* Created initiator matrix
* Applied Kronecker products
* Converted matrix to NetworkX graph

### Graph Analysis

Computed:

* Number of nodes
* Number of edges
* Degree distribution
* Connected components
* Clustering coefficient
* Density
* Shortest paths

### Visualization

Generated:

* Spring Layout
* Circular Layout
* Degree Distribution Histogram

---

# Python Files Created

```text
Day5_6_7_network_science/

├── notes.md
├── exercise_5A_graphs_networkx.py
└── exercise_5B_kronecker_graphs.py
```

---

# Libraries Used

```python
numpy        # Adjacency matrices and numerical operations
networkx     # Graph creation and analysis
matplotlib   # Graph visualization
collections  # deque used for BFS implementation
```

---

# Google Colab Practice

The exercises were executed and tested using Google Colab directly from the GitHub repository.

### Setup Commands

```python
# Clone repository
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

# Navigate to directory
%cd /content/mlsi-summer-internship-2026/Day5_6_7_network_science

# Install required libraries
!pip install numpy networkx matplotlib

# Run Exercise 5A
!python exercise_5A_graphs_networkx.py

# Run Exercise 5B
!python exercise_5B_kronecker_graphs.py
```

---

# Google Colab Notebook

Colab notebook used for implementation and testing:

https://colab.research.google.com/drive/1gCtfJDMqJ5DtfGMxiFVgpdqwJWZYkfEe

---

# Repository Structure

```text
mlsi-summer-internship-2026
│
├── Day1_NumPy
├── Day2_Pandas_Matplotlib
├── Day3_supervised_learning
├── Day4_deep_learning
└── Day5_6_7_network_science
    ├── exercise_5A_graphs_networkx.py
    ├── exercise_5B_kronecker_graphs.py
    └── notes.md
```

---

# Key Learnings

By completing Exercises 5A and 5B, I learned:

* How graphs are represented using adjacency matrices and adjacency lists.
* How to compute degree and neighbors of nodes.
* How BFS works for connectivity and shortest path discovery.
* How connected components are identified.
* How to use NetworkX for graph creation and analysis.
* How to visualize networks using different layouts.
* How Kronecker products generate self-similar networks.
* How graph metrics such as clustering coefficient, density, and path length are computed.
* How to compare custom implementations against NetworkX built-in functions.

---

# Summary

Days 5–7 introduced the complete workflow of network analysis:

1. Represent graphs using adjacency matrices and adjacency lists.
2. Traverse graphs using BFS.
3. Analyze graph properties.
4. Verify results using NetworkX.
5. Generate synthetic networks using Kronecker products.
6. Visualize and interpret graph structures.

These concepts form the foundation of modern Network Science and are widely used in:

* Social Network Analysis
* Communication Networks
* Transportation Networks
* Biological Networks
* Recommendation Systems
* Graph Machine Learning

---

**Course:** MLSI Summer Internship 2026
**Module:** Day 5–7 Network Science & Graph Theory
**Repository:** mlsi-summer-internship-2026
