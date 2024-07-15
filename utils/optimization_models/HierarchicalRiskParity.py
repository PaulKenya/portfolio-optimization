import numpy as np
import pandas as pd
import riskfolio as rp
from utils.performance_calculation import calculate_portfolio_profit


class HierarchicalRiskParity:
    def __init__(self, data: pd.DataFrame, num_assets: int, timestamp: str, timestamp_data: pd.DataFrame, risk: float):
        self.data = data
        self.timestamp_data = timestamp_data
        self.returns = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        self.num_assets = num_assets
        self.timestamp = timestamp
        self.risk = risk
        self.filtered_returns = data
        self.results = []

    def run_hrp(self):
        # Building the portfolio object
        port = rp.HCPortfolio(returns=self.returns)

        model = 'HRP'  # Could be HRP or HERC
        codependence = 'pearson'  # Correlation matrix used to group assets in clusters
        rm = 'MV'  # Risk measure used, this time will be variance
        rf = self.risk  # Risk-free rate
        linkage = 'single'  # Linkage method used to build clusters
        max_k = 4  # Max number of clusters used in two difference gap statistic, only for HERC model
        leaf_order = True  # Consider optimal order of leafs in dendrogram

        weights = port.optimization(model=model,
                                    codependence=codependence,
                                    rm=rm,
                                    rf=rf,
                                    linkage=linkage,
                                    max_k=max_k,
                                    leaf_order=leaf_order)

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-weights.values.flatten())  # Sort weights in descending order
        top_indices = sorted_indices[:self.num_assets]

        # Create optimal weights vector with only the top assets
        num_assets_total = len(self.returns.columns)
        optimal_weights = np.zeros(num_assets_total)
        optimal_weights[top_indices] = weights.values.flatten()[top_indices]

        # Normalize the weights to sum to 1 for the selected assets
        total_weight = np.sum(optimal_weights)
        if total_weight > 0:
            return optimal_weights / total_weight
        else:
            print("----------> Error: No valid weights found.")
            return None

    def optimize(self):
        num_assets_total = len(self.returns.columns)
        if num_assets_total == 0:
            print("----------> Error: No assets available for optimization.")
            return None

        optimal_weights = self.run_hrp()
        if optimal_weights is None:
            print(f"----------> Optimization with Hierarchical Risk Parity did not converge.")
            return self.results

        selected_assets = self.filtered_returns.columns[optimal_weights > 0]
        weights = optimal_weights[optimal_weights > 0]
        profit_percentage = calculate_portfolio_profit(self.timestamp_data, selected_assets, weights)
        weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
        self.results.append({
            'Timestamp': pd.to_datetime(self.timestamp, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': 'HRP',
            'Weights': weights_dict,
            'Profit Percentage': profit_percentage
        })
        print(f"----------> Optimization with Hierarchical Risk Parity completed. Timestamp: {self.timestamp}")
        return self.results
