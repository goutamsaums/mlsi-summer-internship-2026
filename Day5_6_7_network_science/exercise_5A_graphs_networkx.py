"""
Exercise 5A: Graphs as a Data Structure using NetworkX

Topics Implemented:
- Graph Theory Fundamentals
- Nodes and Edges
- Undirected Graphs
- Adjacency Matrix (NumPy)
- Adjacency List
- Degree
- Neighbors
- BFS Traversal
- Paths
- Connected Components
- NetworkX Library
- Graph Visualization

Note: This exercise focuses on UNDIRECTED graphs only.
      Concepts like directed graphs, DFS, Erdős–Rényi,
      Watts-Strogatz, and centrality measures are not
      implemented here (studied separately in theory).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# ==========================================================
# (a) BUILD A SMALL SOCIAL NETWORK GRAPH
# ==========================================================

print("="*70)
print("EXERCISE 5A: GRAPHS AS A DATA STRUCTURE")
print("="*70)

# Social network of 10 people (nodes)
people = [
    "Alice",    # 0
    "Bob",      # 1
    "Charlie",  # 2
    "David",    # 3
    "Emma",     # 4
    "Frank",    # 5
    "Grace",    # 6
    "Helen",    # 7
    "Ivy",      # 8
    "Jack"      # 9
]

n = len(people)  # 10 nodes

# 15 undirected friendship edges
edges = [
    (0, 1),  # Alice-Bob
    (0, 2),  # Alice-Charlie
    (0, 3),  # Alice-David
    (1, 2),  # Bob-Charlie
    (1, 4),  # Bob-Emma
    (2, 5),  # Charlie-Frank
    (3, 4),  # David-Emma
    (3, 6),  # David-Grace
    (4, 5),  # Emma-Frank
    (4, 7),  # Emma-Helen
    (5, 8),  # Frank-Ivy
    (6, 7),  # Grace-Helen
    (6, 9),  # Grace-Jack
    (7, 8),  # Helen-Ivy
    (8, 9)   # Ivy-Jack
]

print(f"\nGraph Created:")
print(f"  Nodes: {n} people")
print(f"  Edges: {len(edges)} friendships")
print(f"  Type: Undirected Graph (friendships are mutual)")

# ==========================================================
# ADJACENCY MATRIX (NumPy)
# ==========================================================

print("\n" + "-"*60)
print("ADJACENCY MATRIX (NumPy)")
print("-"*60)

adj_matrix = np.zeros((n, n), dtype=int)

for u, v in edges:
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1  # Undirected: symmetric matrix

print(adj_matrix)

# ==========================================================
# ADJACENCY LIST (From Scratch)
# ==========================================================

print("\n" + "-"*60)
print("ADJACENCY LIST (Dictionary Representation)")
print("-"*60)

adj_list = {i: [] for i in range(n)}

for u, v in edges:
    adj_list[u].append(v)
    adj_list[v].append(u)

# Display adjacency list with names
for node in sorted(adj_list.keys()):
    neighbors_names = [people[nbr] for nbr in sorted(adj_list[node])]
    print(f"  {people[node]:8s} -> {neighbors_names}")

# ==========================================================
# NETWORKX GRAPH
# ==========================================================

print("\n" + "-"*60)
print("NETWORKX GRAPH")
print("-"*60)

G = nx.Graph()  # Undirected graph
G.add_nodes_from(range(n))
G.add_edges_from(edges)

# Add node attributes (names for visualization)
for i, name in enumerate(people):
    G.nodes[i]['name'] = name

print(f"  Graph type: {type(G).__name__}")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Is directed: {G.is_directed()}")
print(f"  Is weighted: {nx.is_weighted(G)}")

# ==========================================================
# VERIFY ADJACENCY MATRIX == NETWORKX GRAPH
# ==========================================================

print("\n" + "-"*60)
print("VERIFICATION")
print("-"*60)

nx_matrix = nx.to_numpy_array(G, dtype=int)

if np.array_equal(adj_matrix, nx_matrix):
    print("  ✓ NumPy adjacency matrix matches NetworkX graph")
    print("  ✓ Both representations encode the same information")
else:
    print("  ✗ ERROR: Representations do not match")

# ==========================================================
# (b) FUNCTIONS FROM SCRATCH
# ==========================================================

print("\n" + "="*70)
print("(b) FUNCTIONS FROM SCRATCH")
print("="*70)

def compute_degree(matrix, node):
    """
    Compute degree of a node using adjacency matrix.
    Degree = number of edges incident to the node.
    
    Time complexity: O(n) where n = number of nodes
    """
    degree = 0
    for value in matrix[node]:
        degree += value
    return degree

def find_neighbors(matrix, node):
    """
    Find all neighbors of a node using adjacency matrix.
    Neighbors = nodes directly connected to this node.
    
    Time complexity: O(n) where n = number of nodes
    """
    neighbors = []
    for j in range(len(matrix)):
        if matrix[node][j] == 1:
            neighbors.append(j)
    return neighbors

def bfs_connected(matrix, start, target):
    """
    Check if two nodes are connected using BFS traversal.
    BFS (Breadth-First Search) explores level by level.
    
    Time complexity: O(V + E) where V = vertices, E = edges
    """
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

def bfs_find_path(matrix, start, target):
    """
    Find the shortest path between nodes using BFS.
    Returns list of nodes representing the path, or None if no path.
    """
    if start == target:
        return [start]
    
    visited = set([start])
    queue = deque([[start]])
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        
        for neighbor in find_neighbors(matrix, current):
            if neighbor == target:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    
    return None

def find_connected_components(matrix):
    """
    Find all connected components using BFS.
    A connected component is a maximal set of mutually reachable nodes.
    """
    n_nodes = len(matrix)
    visited = set()
    components = []
    
    for node in range(n_nodes):
        if node not in visited:
            # BFS to find entire component
            component = []
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in find_neighbors(matrix, current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components

# ==========================================================
# TEST CUSTOM FUNCTIONS
# ==========================================================

print("\n" + "-"*60)
print("TESTING CUSTOM FUNCTIONS")
print("-"*60)

# Test 1: Degrees
print("\n1. Node Degrees:")
for i in range(n):
    deg = compute_degree(adj_matrix, i)
    print(f"   {people[i]}: {deg}")

# Test 2: Neighbors (sample)
print("\n2. Node Neighbors (sample):")
sample_nodes = [0, 4, 9]  # Alice, Emma, Jack
for i in sample_nodes:
    nbrs = find_neighbors(adj_matrix, i)
    nbr_names = [people[x] for x in nbrs]
    print(f"   {people[i]}: {nbr_names}")

# Test 3: BFS Connectivity
print("\n3. BFS Connectivity:")
test_pairs = [
    (0, 9, "Alice → Jack"),
    (1, 8, "Bob → Ivy"),
    (2, 7, "Charlie → Helen"),
    (0, 0, "Alice → Alice")
]
for s, t, label in test_pairs:
    result = bfs_connected(adj_matrix, s, t)
    print(f"   {label}: {result}")

# Test 4: BFS Path
print("\n4. BFS Shortest Path:")
path = bfs_find_path(adj_matrix, 0, 9)
if path:
    path_names = [people[x] for x in path]
    print(f"   Alice → Jack: {' → '.join(path_names)}")
    print(f"   Path length: {len(path)-1} steps")

# Test 5: Connected Components
print("\n5. Connected Components:")
components = find_connected_components(adj_matrix)
print(f"   Number of components: {len(components)}")
for i, comp in enumerate(components, 1):
    comp_names = [people[x] for x in comp]
    print(f"   Component {i}: {comp_names}")

# ==========================================================
# (c) VERIFY AGAINST NETWORKX BUILT-INS
# ==========================================================

print("\n" + "="*70)
print("(c) VERIFICATION WITH NETWORKX")
print("="*70)

all_match = True

# 1. Degree verification
print("\n1. Degree Verification:")
for node in range(n):
    custom = compute_degree(adj_matrix, node)
    nx_deg = G.degree[node]
    match = (custom == nx_deg)
    status = "✓" if match else "✗"
    print(f"   {people[node]}: Custom={custom}, NetworkX={nx_deg} → {status}")
    if not match:
        all_match = False

# 2. Neighbor verification
print("\n2. Neighbor Verification:")
for node in sample_nodes:
    custom = set(find_neighbors(adj_matrix, node))
    nx_nbrs = set(G.neighbors(node))
    match = (custom == nx_nbrs)
    status = "✓" if match else "✗"
    print(f"   {people[node]}: {match} → {status}")
    if not match:
        all_match = False

# 3. Connectivity verification (all pairs)
print("\n3. Connectivity Verification:")
all_pairs_match = True
for s in range(n):
    for t in range(n):
        custom = bfs_connected(adj_matrix, s, t)
        nx_result = nx.has_path(G, s, t)
        if custom != nx_result:
            all_pairs_match = False
            break
    if not all_pairs_match:
        break

print(f"   All node pairs match: {all_pairs_match} → {'✓' if all_pairs_match else '✗'}")
if not all_pairs_match:
    all_match = False

# 4. Components verification (compare sizes)
print("\n4. Components Verification:")
custom_comps = find_connected_components(adj_matrix)
nx_comps = list(nx.connected_components(G))

custom_sizes = sorted(len(c) for c in custom_comps)
nx_sizes = sorted(len(c) for c in nx_comps)

match = (custom_sizes == nx_sizes)
status = "✓" if match else "✗"
print(f"   Component sizes: Custom={custom_sizes}, NetworkX={nx_sizes} → {status}")
if not match:
    all_match = False

# 5. Path verification (sample)
print("\n5. Path Verification (sample):")
custom_path = bfs_find_path(adj_matrix, 0, 9)
nx_path = nx.shortest_path(G, 0, 9) if nx.has_path(G, 0, 9) else None
match = (custom_path == nx_path)
status = "✓" if match else "✗"
path_str = ' → '.join([people[x] for x in custom_path]) if custom_path else 'None'
print(f"   Alice → Jack path: {path_str}")
print(f"   Paths match: {status}")

# Final result
print("\n" + "-"*60)
if all_match:
    print("✓✓✓ ALL VERIFICATIONS PASSED! ✓✓✓")
    print("Custom functions match NetworkX built-ins perfectly.")
else:
    print("✗✗✗ VERIFICATION FAILED - Check implementations ✗✗✗")
print("-"*60)

# ==========================================================
# BONUS: GRAPH ANALYSIS
# ==========================================================

print("\n" + "="*70)
print("BONUS: GRAPH ANALYSIS")
print("="*70)

# Basic statistics
print(f"\nBasic Graph Statistics:")
print(f"  Number of nodes: {G.number_of_nodes()}")
print(f"  Number of edges: {G.number_of_edges()}")
print(f"  Graph density: {nx.density(G):.3f}")
print(f"  Is connected: {nx.is_connected(G)}")

# Degree distribution analysis
degrees = [d for n, d in G.degree()]
print(f"\nDegree Distribution Analysis:")
print(f"  Min degree: {min(degrees)}")
print(f"  Max degree: {max(degrees)}")
print(f"  Mean degree: {np.mean(degrees):.2f}")
print(f"  Standard deviation: {np.std(degrees):.2f}")
print(f"  Max/Mean ratio: {max(degrees)/np.mean(degrees):.2f}")

# Clustering analysis
print(f"\nClustering Analysis:")
print(f"  Average clustering coefficient: {nx.average_clustering(G):.3f}")
print(f"  Global clustering (transitivity): {nx.transitivity(G):.3f}")

# Shortest path analysis (only if connected)
if nx.is_connected(G):
    print(f"\nShortest Path Analysis:")
    print(f"  Graph diameter: {nx.diameter(G)}")
    print(f"  Average path length: {nx.average_shortest_path_length(G):.3f}")

# ==========================================================
# (d) VISUALIZE GRAPH
# ==========================================================

print("\n" + "="*70)
print("(d) GRAPH VISUALIZATION")
print("="*70)
print("\nGenerating visualizations...")

# Create figure with two layouts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Layout 1: Spring layout (force-directed)
pos1 = nx.spring_layout(G, seed=42, k=1.5)
labels = {i: people[i] for i in range(n)}

nx.draw(G, pos1, ax=ax1, with_labels=True, labels=labels,
        node_color='lightblue', node_size=1200,
        edge_color='gray', font_size=9, font_weight='bold')
ax1.set_title("Social Network - Spring Layout", fontsize=12, fontweight='bold')

# Layout 2: Circular layout
pos2 = nx.circular_layout(G)
nx.draw(G, pos2, ax=ax2, with_labels=True, labels=labels,
        node_color='lightgreen', node_size=1200,
        edge_color='darkgray', font_size=9, font_weight='bold')
ax2.set_title("Social Network - Circular Layout", fontsize=12, fontweight='bold')

plt.suptitle(f"Social Network Graph\n{G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Degree distribution histogram
fig2, ax = plt.subplots(figsize=(10, 5))

degree_counts = {}
for d in degrees:
    degree_counts[d] = degree_counts.get(d, 0) + 1

degrees_plot = sorted(degree_counts.keys())
counts_plot = [degree_counts[d] for d in degrees_plot]

ax.bar(degrees_plot, counts_plot, width=0.8, 
       edgecolor='black', alpha=0.7, color='skyblue')
ax.set_xlabel('Degree', fontsize=12)
ax.set_ylabel('Number of Nodes', fontsize=12)
ax.set_title('Degree Distribution of Social Network', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add statistics annotation
stats_text = f"Mean: {np.mean(degrees):.2f}\nMax: {max(degrees)}\nMin: {min(degrees)}"
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n✓ Visualization complete - 2 layouts + degree histogram")

# ==========================================================
# SUMMARY OF IMPLEMENTED CONCEPTS
# ==========================================================

print("\n" + "="*70)
print("EXERCISE 5A COMPLETE - IMPLEMENTED CONCEPTS")
print("="*70)

concepts = [
    "✓ Graph as a data structure",
    "✓ Nodes (10 vertices - people)",
    "✓ Edges (15 connections - friendships)",
    "✓ Undirected graph",
    "✓ Adjacency matrix (NumPy)",
    "✓ Adjacency list (dictionary)",
    "✓ Degree (custom function from scratch)",
    "✓ Neighbors (custom function from scratch)",
    "✓ BFS connectivity (custom function from scratch)",
    "✓ BFS shortest path",
    "✓ Connected components (custom function)",
    "✓ NetworkX graph creation",
    "✓ Verification against NetworkX built-ins",
    "✓ Graph visualization (2 layouts + histogram)"
]

for concept in concepts:
    print(concept)

print("\n" + "="*70)
print("✓✓✓ ASSIGNMENT COMPLETE - ALL TASKS (a-d) FINISHED ✓✓✓")
print("="*70)
