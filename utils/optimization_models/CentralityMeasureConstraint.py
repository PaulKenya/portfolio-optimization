import numpy as np
import pandas as pd
import cvxpy as cp

from utils.centrality_measures import calculate_centrality_measures
from utils.optimization_models.Graph import Graph
from utils.performance_calculation import calculate_portfolio_profit


class CentralityMeasureConstraint:
    def __init__(self, data: pd.DataFrame, graph: Graph, desired_avg_centrality: float, num_assets: int, timestamp: str):
        self.returns = data.mean().values
        self.covariance_matrix = data.cov().values
        self.data = data
        self.graph = graph
        self.num_assets = num_assets
        self.desired_avg_centrality = desired_avg_centrality
        self.timestamp = timestamp
        self.results = []

    def run_centrality_optimization(self, centrality_measure):
        centrality_vector = np.array([centrality_measure[node] for node in self.data.columns])

        num_assets_total = len(self.returns)
        x = cp.Variable(num_assets_total)

        # Define the optimization problem
        objective = cp.Maximize(self.returns @ x - cp.quad_form(x, self.covariance_matrix))
        centrality_constraint_lower = centrality_vector @ x >= self.desired_avg_centrality * 0.95
        centrality_constraint_upper = centrality_vector @ x <= self.desired_avg_centrality * 1.05
        constraints = [
            cp.sum(x) == 1,  # Weights sum to 1
            x >= 0,          # No short selling
            centrality_constraint_lower,
            centrality_constraint_upper
        ]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS)  # You can adjust the solver if needed
        except Exception as e:
            print(f"----------> Optimization problem encountered an error: {e}")
            return None

        if prob.status != cp.OPTIMAL:
            return None

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
        top_indices = sorted_indices[:self.num_assets]

        # Create optimal weights vector with only the top assets
        optimal_weights = np.zeros(num_assets_total)
        optimal_weights[top_indices] = x.value[top_indices]

        # Normalize the weights to sum to 1 for the selected assets
        total_weight = np.sum(optimal_weights)
        if total_weight > 0:
            return optimal_weights / total_weight
        else:
            print("----------> Error: No valid weights found.")
            return None

    def optimize(self):
        if len(self.returns) == 0:
            print("----------> Error: No assets available for optimization.")
            return self.results

        for graph_type in ['mst', 'tmfg']:
            if graph_type == 'mst':
                graph = self.graph.mst()
            else:
                graph = self.graph.tmfg()

            degree_centrality, eigenvector_centrality, subgraph_centrality = calculate_centrality_measures(graph)
            centrality_measures = {
                f"{graph_type.upper()} Degree Centrality": degree_centrality,
                f"{graph_type.upper()} Eigenvector Centrality": eigenvector_centrality,
                f"{graph_type.upper()} Subgraph Centrality": subgraph_centrality
            }

            for centrality_name, centrality_measure in centrality_measures.items():
                if centrality_measure:
                    optimal_weights = self.run_centrality_optimization(centrality_measure)

                    if optimal_weights is None:
                        print(f"----------> Optimization with {centrality_name} did not converge.")
                        continue

                    selected_assets = self.data.columns[optimal_weights > 0]
                    weights = optimal_weights[optimal_weights > 0]
                    profit_percentage = calculate_portfolio_profit(self.data, selected_assets, weights)
                    weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
                    self.results.append({
                        'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        'Optimization Type': centrality_name,
                        'Weights': weights_dict,
                        'Profit Percentage': profit_percentage
                    })

                    print(f"----------> Optimization with {centrality_name} completed. Timestamp: {self.timestamp}")

        return self.results
