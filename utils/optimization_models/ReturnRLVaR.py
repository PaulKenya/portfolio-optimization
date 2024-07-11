import numpy as np
import pandas as pd
import cvxpy as cp

from utils.performance_calculation import calculate_portfolio_profit

class ReturnRLVaR:
    def __init__(self, data: pd.DataFrame, num_assets: int, timestamp: str,
                 target_return: float = 0.02, alpha: float = 0.05, kappa: float = 0.3):
        """
        Initializes the ReturnRLVaR class.

        Parameters:
        - data: pd.DataFrame containing asset returns
        - num_assets: int, number of assets to include in the optimized portfolio
        - timestamp: str, timestamp for the optimization run
        - target_return: float, target return for the portfolio (default: 0.02)
        - alpha: float, confidence level for VaR calculation (default: 0.05)
        - kappa: float, deformation parameter for RLVaR (default: 0.3)
        """
        self.data = data
        self.returns = data.mean().values
        self.covariance_matrix = data.cov().values
        self.num_assets = num_assets
        self.target_return = target_return
        self.alpha = alpha
        self.kappa = kappa
        self.timestamp = timestamp
        self.results = []

    def run_rlvar(self):
        num_assets_total = len(self.returns)
        num_samples = self.data.shape[0]

        x = cp.Variable(num_assets_total)
        t = cp.Variable()
        z = cp.Variable()
        psi = cp.Variable(num_samples)
        theta = cp.Variable(num_samples)
        epsilon = cp.Variable(num_samples)
        omega = cp.Variable(num_samples)

        # Define the RLVaR constraints based on the primal formulation
        constraints = [
            self.returns @ x >= self.target_return,
            cp.sum(x) == 1,
            x >= 0,
            z >= 0,
            -self.data.values @ x - t + epsilon + omega <= 0,
            z * (1 + self.kappa) / (2 * self.kappa) >= cp.abs(psi * (1 + self.kappa) / self.kappa) + epsilon - 1e-6,
            omega * (1 / (1 - self.kappa)) >= cp.abs(theta * (1 / self.kappa)) + z * (1 / (2 * self.kappa)) - 1e-6
        ]

        ln_kappa = (1 / self.kappa) * np.log(1 / (self.alpha * num_samples))
        objective = cp.Minimize(t + z * ln_kappa + cp.sum(psi + theta))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception as e:
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
        profit_percentage = calculate_portfolio_profit(self.data, selected_assets, weights)
        weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
        self.results.append({
            'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': 'Return RLVaR Optimization',
            'Weights': weights_dict,
            'Profit Percentage': profit_percentage
        })
        print(f"----------> Optimization with RLVaR completed. Timestamp: {self.timestamp}")
        return self.results
