import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from utils.performance_calculation import calculate_portfolio_profit


class HierarchicalRiskParity:
    def __init__(self, data: pd.DataFrame, num_assets: int, timestamp: str, timestamp_data: pd.DataFrame, method: str = 'single', w_min: float = 0.0, w_max: float = 1.0, lam: float = 1.0):
        self.data = data
        self.timestamp_data = timestamp_data
        self.returns = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        self.covariance_matrix = self.returns.cov().values
        self.num_assets = num_assets
        self.timestamp = timestamp
        self.method = method
        self.w_min = w_min
        self.w_max = w_max
        self.lam = lam
        self.results = []

    def get_ivp(self, Sigma):
        if Sigma.ndim == 0:
            return np.array([1.0])
        ivp = np.where(np.diag(Sigma) == 0, 0, 1.0 / np.diag(Sigma))
        return ivp / np.sum(ivp)

    def get_cluster_var(self, cluster_idx):
        Sigma_ = self.covariance_matrix[np.ix_(cluster_idx, cluster_idx)]
        w_ = self.get_ivp(Sigma_)
        return np.dot(w_.T, np.dot(Sigma_, w_))

    def bisect_clusters(self, clusters):
        clusters_new = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            n = len(cl) // 2
            clusters_new.append(cl[:n])
            clusters_new.append(cl[n:])
        return clusters_new

    def get_rec_bipart(self, sorted_idx, w_min, w_max, lam):
        N = len(sorted_idx)
        w = np.ones(N)
        clusters = self.bisect_clusters([sorted_idx])

        while len(clusters) > 0:
            new_clusters = []
            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    new_clusters.append(clusters[i])
                    continue

                cl0 = clusters[i]
                cl1 = clusters[i + 1]

                cl_var0 = lam * self.get_cluster_var(cl0)
                cl_var1 = self.get_cluster_var(cl1)
                alpha = 1 - cl_var0 / (cl_var0 + cl_var1)

                alpha = min(np.sum(w_max[cl0]) / w[cl0[0]], max(np.sum(w_min[cl0]) / w[cl0[0]], alpha))
                alpha = 1 - min(np.sum(w_max[cl1]) / w[cl1[0]], max(np.sum(w_min[cl1]) / w[cl1[0]], 1 - alpha))

                w[cl0] *= alpha
                w[cl1] *= (1 - alpha)

            clusters = self.bisect_clusters(new_clusters)

        return w

    def optimize(self):
        if self.returns.empty:
            print("----------> Error: No returns available for optimization.")
            return None

        rho = np.corrcoef(self.returns.T)
        distance = np.sqrt((1 - rho) / 2)

        # Ensure that the distance matrix contains only finite values
        distance = np.nan_to_num(distance, nan=1.0, posinf=1.0, neginf=1.0)

        # Convert the distance matrix to condensed form
        condensed_distance = pdist(distance)

        if self.method == 'divisive':
            hcluster = sch.dendrogram(sch.linkage(condensed_distance, method='ward'), no_plot=True)
        else:
            hcluster = sch.dendrogram(sch.linkage(condensed_distance, method=self.method), no_plot=True)

        sorted_idx = hcluster['leaves']

        w_min = np.repeat(self.w_min, self.covariance_matrix.shape[1])
        w_max = np.repeat(self.w_max, self.covariance_matrix.shape[1])

        w = self.get_rec_bipart(sorted_idx, w_min, w_max, self.lam)

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-w)  # Sort weights in descending order
        top_indices = sorted_indices[:self.num_assets]

        # Create optimal weights vector with only the top assets
        optimal_weights = np.zeros_like(w)
        optimal_weights[top_indices] = w[top_indices]

        # Normalize the weights to sum to 1 for the selected assets
        total_weight = np.sum(optimal_weights)
        if total_weight > 0:
            optimal_weights /= total_weight
        else:
            print("----------> Error: No valid weights found.")
            return None

        selected_assets = self.data.columns[optimal_weights > 0]
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
