"""
Exercise 5B: Kronecker Graphs using NumPy and NetworkX

Topics Covered:
- Kronecker Product
- Adjacency Matrix
- Graph Generation
- Degree Distribution
- Connected Components
- NetworkX Analysis
- Graph Visualization
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================================
# STEP 1: CREATE INITIATOR MATRIX
# ==========================================================

K1 = np.array([
    [1, 1],
    [1, 0]
])

print("\nInitiator Matrix (K1)")
print(K1)

# ==========================================================
# STEP 2: GENERATE KRONECKER GRAPH
# ==========================================================

K2 = np.kron(K1, K1)

print("\nKronecker Expansion (K2)")
print(K2)

K3 = np.kron(K2, K1)

print("\nKronecker Expansion (K3)")
print(K3)

# ==========================================================
# STEP 3: CREATE NETWORKX GRAPH
# ==========================================================

G = nx.from_numpy_array(K3)

print("\nGraph Created Successfully")

# ==========================================================
# STEP 4: BASIC GRAPH INFORMATION
# ==========================================================

print("\nBasic Graph Information")

print("Number of Nodes :", G.number_of_nodes())
print("Number of Edges :", G.number_of_edges())

# ==========================================================
# STEP 5: NODE DEGREES
# ==========================================================

print("\nNode Degrees")

for node, degree in G.degree():
    print(f"Node {node}: Degree = {degree}")

# ==========================================================
# STEP 6: DEGREE DISTRIBUTION
# ==========================================================

degrees = [degree for _, degree in G.degree()]

print("\nDegree Distribution")

unique_degrees = sorted(set(degrees))

for d in unique_degrees:
    count = degrees.count(d)
    print(f"Degree {d}: {count} nodes")

# ==========================================================
# STEP 7: CONNECTED COMPONENTS
# ==========================================================

components = list(nx.connected_components(G))

print("\nConnected Components")

for i, component in enumerate(components, start=1):
    print(f"Component {i}: {sorted(component)}")

print("\nNumber of Connected Components:",
      nx.number_connected_components(G))

# ==========================================================
# STEP 8: SHORTEST PATH EXAMPLE
# ==========================================================

source = 0
target = max(G.nodes)

print("\nShortest Path")

if nx.has_path(G, source, target):

    path = nx.shortest_path(G, source, target)

    print(f"Path from {source} to {target}:")
    print(path)

else:

    print("No path exists.")

# ==========================================================
# STEP 9: CLUSTERING COEFFICIENT
# ==========================================================

print("\nClustering Coefficients")

clustering = nx.clustering(G)

for node, coeff in clustering.items():
    print(f"Node {node}: {coeff:.3f}")

print(
    "\nAverage Clustering Coefficient:",
    round(nx.average_clustering(G), 3)
)

# ==========================================================
# STEP 10: DENSITY
# ==========================================================

print("\nGraph Density")

print(
    round(nx.density(G), 4)
)

# ==========================================================
# STEP 11: VISUALIZATION
# ==========================================================

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightgreen",
    node_size=1000,
    edge_color="gray",
    font_size=9,
    width=2
)

plt.title(
    "Kronecker Graph Generated from K1"
)

plt.show()

# ==========================================================
# STEP 12: DEGREE HISTOGRAM
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    degrees,
    bins=range(
        min(degrees),
        max(degrees) + 2
    ),
    align="left"
)

plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")

plt.grid(True)

plt.show()
