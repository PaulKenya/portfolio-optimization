import numpy as np
import networkx as nx
from itertools import combinations

import pandas as pd


def calculate_mst(correlation_matrix: pd.DataFrame) -> nx.Graph:
    print("----------> calculate_mst: Starting MST calculation.")
    graph = nx.Graph()

    # Fill NaN values with a default value (e.g., 0)
    correlation_matrix = correlation_matrix.fillna(0)

    # Add edges to the graph
    print("----------> calculate_mst: Adding edges to the graph.")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            weight = 1 - correlation_matrix.iloc[i, j]
            graph.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=weight)

    # Calculate MST using Kruskal's algorithm
    mst = nx.minimum_spanning_tree(graph, weight='weight')
    return mst


def calculate_tmfg(correlation_matrix: pd.DataFrame) -> nx.Graph:
    print("----------> calculate_tmfg: Starting TMFG calculation.")

    n = len(correlation_matrix)
    indices = np.arange(n)
    edges = []

    # Create a fully connected subgraph (clique) with the highest sum of correlations
    initial_clique = None
    max_weight = -np.inf

    print("----------> calculate_tmfg: Finding initial clique with the highest sum of correlations.")
    for clique in combinations(indices, 4):
        weight = np.sum(correlation_matrix[np.ix_(clique, clique)])
        if weight > max_weight:
            max_weight = weight
            initial_clique = clique

    # Check if initial_clique is found
    if initial_clique is None:
        raise ValueError("No valid initial clique found. Check the correlation matrix.")

    print(f"----------> calculate_tmfg: Initial clique found: {initial_clique} with weight: {max_weight}")

    # Add the initial clique to the graph
    tmfg = nx.Graph()
    tmfg.add_nodes_from(initial_clique)

    for i, j in combinations(initial_clique, 2):
        tmfg.add_edge(i, j, weight=1 - correlation_matrix[i, j])
        edges.append((i, j, 1 - correlation_matrix[i, j]))

    print(f"----------> calculate_tmfg: Initial clique added to the graph with nodes: {initial_clique}")

    remaining_nodes = set(indices) - set(initial_clique)

    # Iteratively add the remaining nodes
    while remaining_nodes:
        max_gain = -np.inf
        best_node = None
        best_face = None

        for node in remaining_nodes:
            for face in nx.cycle_basis(tmfg):
                if len(face) == 3:  # Ensure it's a triangle
                    gain = np.sum(correlation_matrix[node, face])
                    if gain > max_gain:
                        max_gain = gain
                        best_node = node
                        best_face = face

        if best_node is None or best_face is None:
            raise ValueError("Failed to find the best node or face. Check the correlation matrix and graph structure.")

        print(f"----------> calculate_tmfg: Adding node {best_node} to the graph, connecting to face {best_face}.")

        remaining_nodes.remove(best_node)
        for i, j in combinations(best_face, 2):
            tmfg.add_edge(i, j, weight=1 - correlation_matrix[i, j])
            edges.append((i, j, 1 - correlation_matrix[i, j]))

        for i in best_face:
            tmfg.add_edge(i, best_node, weight=1 - correlation_matrix[i, best_node])
            edges.append((i, best_node, 1 - correlation_matrix[i, best_node]))

    print("----------> calculate_tmfg: TMFG construction completed.")
    return tmfg


def create_tmfg_approx(correlation_matrix: pd.DataFrame, k: int = 3) -> nx.Graph:
    print("----------> create_tmfg_approx: Starting TMFG approximation calculation.")
    distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))

    # Create a fully connected graph with the distances as weights
    graph = nx.Graph()
    print("----------> create_tmfg_approx: Adding edges to the fully connected graph.")

    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            graph.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=distance_matrix.iloc[i, j])
            print(f"----------> create_tmfg_approx: Edge added between {correlation_matrix.index[i]} and {correlation_matrix.columns[j]} with weight {distance_matrix.iloc[i, j]}")

    print("----------> create_tmfg_approx: All edges added to the fully connected graph.")
    print("----------> create_tmfg_approx: Starting K-nearest neighbors filtering.")

    # Use a filtering method to approximate TMFG (K-nearest neighbors approach)
    graph_tmfg = nx.Graph()

    for node in graph.nodes():
        neighbors = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[:k]
        for neighbor in neighbors:
            graph_tmfg.add_edge(node, neighbor[0], weight=neighbor[1]['weight'])
            print(f"----------> create_tmfg_approx: Edge added between {node} and {neighbor[0]} with weight {neighbor[1]['weight']}")

    print("----------> create_tmfg_approx: TMFG approximation completed.")
    return graph_tmfg

