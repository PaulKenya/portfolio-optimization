import traceback
import numpy as np
import pandas as pd
import cvxpy as cp
from utils.performance_calculation import calculate_portfolio_profit


class ReturnRLVaR:
    def __init__(self, data: pd.DataFrame, num_assets: int, timestamp: str, timestamp_data: pd.DataFrame,
                 target_return: float = 0.02):
        self.data = data
        self.timestamp_data = timestamp_data
        self.returns = data.mean().values
        self.covariance_matrix = data.cov().values
        self.num_assets = num_assets
        self.target_return = target_return
        self.timestamp = timestamp
        self.results = []

    def run_rlvar(self):
        num_assets_total = len(self.returns)

        x = cp.Variable(num_assets_total)

        objective = cp.Minimize(cp.quad_form(x, self.covariance_matrix))
        constraints = [
            self.returns @ x >= self.target_return,  # Ensuring the target return constraint is met
            cp.sum(x) == 1,  # Ensuring the weights sum to 1
            x >= 0  # Ensuring no short selling
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.MOSEK)
        except Exception as e:
            traceback.print_exc()
            print(f"----------> Optimization problem encountered an error: {e}")
            return None

        if prob.status != cp.OPTIMAL:
            print("----------> Optimization problem did not converge.")
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
        num_assets_total = len(self.returns)
        if num_assets_total == 0:
            print("----------> Error: No assets available for optimization.")
            return None

        optimal_weights = self.run_rlvar()
        if optimal_weights is None:
            print(f"----------> Optimization with RLVaR did not converge.")
            return self.results

        selected_assets = self.data.columns[optimal_weights > 0]
        weights = optimal_weights[optimal_weights > 0]
        profit_percentage = calculate_portfolio_profit(self.timestamp_data, selected_assets, weights)
        weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
        self.results.append({
            'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': 'RLVaR',
            'Weights': weights_dict,
            'Profit Percentage': profit_percentage
        })
        print(f"----------> Optimization with RLVaR completed. Timestamp: {self.timestamp}")
        return self.results
