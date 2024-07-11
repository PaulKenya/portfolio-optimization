import networkx as nx
import numpy as np


def calculate_centrality_measures(graph: nx.Graph):
    degree_centrality = {}
    eigenvector_centrality = {}
    subgraph_centrality = {}

    try:
        degree_centrality = nx.degree_centrality(graph)
    except Exception as e:
        print(f"Error calculating degree centrality: {e}")

    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality did not converge. Using a fallback initial guess.")
        # Using a fallback initial guess
        initial_guess = {node: 1 for node in graph}
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06, nstart=initial_guess)
        except nx.PowerIterationFailedConvergence as e:
            print(f"Error: Eigenvector centrality failed to converge with fallback initial guess: {e}")
    except Exception as e:
        print(f"Error calculating eigenvector centrality: {e}")

    try:
        subgraph_centrality = nx.subgraph_centrality(graph)
    except Exception as e:
        print(f"Error calculating subgraph centrality: {e}")

    return degree_centrality, eigenvector_centrality, subgraph_centrality


def average_centrality_measure(weights, centrality):
    return np.dot(weights, list(centrality.values()))


def percentage_invested_in_connected_assets(weights, adjacency_matrix):
    connection_matrix = adjacency_matrix @ adjacency_matrix
    connected_investments = np.dot(weights.T, connection_matrix @ weights)
    total_investment = np.dot(weights.T, weights)
    return connected_investments / total_investment


def map_risk_to_centrality(risk_tolerance):
    # Define risk tolerance levels
    risk_levels = {
        "high": 0.90,
        "moderate": 0.50,
        "low": 0.10
    }
    if risk_tolerance not in risk_levels:
        raise ValueError("Invalid risk tolerance level. Choose from 'high', 'moderate', or 'low'.")
    target_centrality = risk_levels[risk_tolerance]
    return target_centrality
