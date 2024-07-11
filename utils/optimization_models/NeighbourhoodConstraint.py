import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cp

from utils.optimization_models.Graph import Graph
from utils.performance_calculation import calculate_portfolio_profit


class NeighbourhoodConstraintMIP:
    def __init__(self, data: pd.DataFrame, graph: Graph, path_length: int, num_assets: int, timestamp: str):
        self.returns = data.mean().values
        self.covariance_matrix = data.cov().values
        self.data = data
        self.graph = graph
        self.num_assets = num_assets
        self.path_length = path_length
        self.timestamp = timestamp
        self.results = []

    def construct_connection_matrix(self, graph: nx.Graph):
        adjacency_matrix = nx.to_numpy_array(graph)
        connection_matrix = np.linalg.matrix_power(adjacency_matrix, self.path_length)
        return connection_matrix

    def run_neighbourhood_optimization(self, graph: nx.Graph):
        num_assets_total = len(self.returns)
        x = cp.Variable(num_assets_total)
        y = cp.Variable(num_assets_total, boolean=True)

        connection_matrix = self.construct_connection_matrix(graph)
        identity_matrix = np.eye(num_assets_total)

        # Define the optimization problem
        objective = cp.Maximize(self.returns @ x - cp.quad_form(x, self.covariance_matrix))
        constraints = [
            cp.sum(x) == 1,  # Weights sum to 1
            x >= 0,          # No short selling
            (connection_matrix + identity_matrix) @ y <= 1,  # Adjusted Neighborhood constraint
            x <= y,          # Upper bound constraint
            x >= 0.005 * y   # Relaxed Lower bound constraint
        ]

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS_BB)
        except Exception as e:
            print(f"----------> Optimization problem encountered an error: {e}")
            return None

        if prob.status != cp.OPTIMAL:
            print("----------> Optimization problem did not converge.")
            return None

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
        top_indices = sorted_indices[:self.num_assets]

        optimal_weights = np.zeros(num_assets_total)
        optimal_weights[top_indices] = x.value[top_indices]
        return optimal_weights

    def optimize(self):
        if len(self.returns) == 0:
            print("----------> Error: No assets available for optimization.")
            return self.results

        for graph_type in ['mst', 'tmfg']:
            if graph_type == 'mst':
                graph = self.graph.mst()
            else:
                graph = self.graph.tmfg()

            optimal_weights = self.run_neighbourhood_optimization(graph)
            if optimal_weights is None:
                print(f"----------> Optimization with {graph_type.upper()} Neighborhood Constraint did not converge.")
                continue

            selected_assets = self.data.columns[optimal_weights > 0]
            weights = optimal_weights[optimal_weights > 0]
            profit_percentage = calculate_portfolio_profit(self.data, selected_assets, weights)
            weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
            self.results.append({
                'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                'Optimization Type': f'{graph_type.upper()} Neighborhood Constraint MIP',
                'Weights': weights_dict,
                'Profit Percentage': profit_percentage
            })

            print(f"----------> Optimization with {graph_type.upper()} Neighborhood Constraint completed. Timestamp: {self.timestamp}")

        return self.results


class NeighbourhoodConstraintSDP:
    def __init__(self, data: pd.DataFrame, graph: Graph, path_length: int, num_assets: int, timestamp: str):
        self.returns = data.mean().values
        self.covariance_matrix = data.cov().values
        self.data = data
        self.graph = graph
        self.num_assets = num_assets
        self.path_length = path_length
        self.timestamp = timestamp
        self.results = []

    def construct_connection_matrix(self, graph: nx.Graph):
        adjacency_matrix = nx.to_numpy_array(graph)
        connection_matrix = np.linalg.matrix_power(adjacency_matrix, self.path_length)
        return connection_matrix

    def run_neighbourhood_optimization(self, graph: nx.Graph):
        num_assets_total = len(self.returns)
        x = cp.Variable(num_assets_total)
        X = cp.Variable((num_assets_total, num_assets_total), symmetric=True)

        connection_matrix = self.construct_connection_matrix(graph)

        # Define the optimization problem
        objective = cp.Minimize(cp.trace(self.covariance_matrix @ X))
        constraints = [
            cp.bmat([[X, x[:, None]], [x[None, :], np.array([[1]])]]) >> 0,   # Semidefinite constraint
            X == X.T,                          # Symmetry constraint
            cp.multiply(connection_matrix, X) >= -1e-6,  # Neighborhood constraint
            cp.sum(x) == 1,                    # Weights sum to 1
            x >= 0                             # No short selling
        ]

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS)
        except Exception as e:
            print(f"----------> Optimization problem encountered an error: {e}")
            return None

        if prob.status != cp.OPTIMAL:
            print("----------> Optimization problem did not converge.")
            return None

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
        top_indices = sorted_indices[:self.num_assets]

        optimal_weights = np.zeros(num_assets_total)
        optimal_weights[top_indices] = x.value[top_indices]
        return optimal_weights

    def optimize(self):
        if len(self.returns) == 0:
            print("----------> Error: No assets available for optimization.")
            return self.results

        for graph_type in ['mst', 'tmfg']:
            if graph_type == 'mst':
                graph = self.graph.mst()
            else:
                graph = self.graph.tmfg()

            optimal_weights = self.run_neighbourhood_optimization(graph)
            if optimal_weights is None:
                print(f"----------> Optimization with {graph_type.upper()} Neighborhood Constraint did not converge.")
                continue

            selected_assets = self.data.columns[optimal_weights > 0]
            weights = optimal_weights[optimal_weights > 0]
            profit_percentage = calculate_portfolio_profit(self.data, selected_assets, weights)
            weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
            self.results.append({
                'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                'Optimization Type': f'{graph_type.upper()} Neighborhood Constraint SDP',
                'Weights': weights_dict,
                'Profit Percentage': profit_percentage
            })

            print(f"----------> Optimization with {graph_type.upper()} Neighborhood Constraint completed. Timestamp: {self.timestamp}")

        return self.results
