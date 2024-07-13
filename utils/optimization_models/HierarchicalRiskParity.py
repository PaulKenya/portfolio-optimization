import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from utils.performance_calculation import calculate_portfolio_profit

class HierarchicalRiskParity:
    def __init__(self, data: pd.DataFrame, method='single', w_min=None, w_max=None, lam=1):
        self.data = data
        self.asset_returns = data
        self.covariance_matrix = data.cov().values
        self.method = method
        self.w_min = w_min if w_min is not None else np.zeros(self.covariance_matrix.shape[0])
        self.w_max = w_max if w_max is not None else np.ones(self.covariance_matrix.shape[0])
        self.lam = lam
        self.results = []

    def get_ivp(self):
        if self.covariance_matrix.ndim == 1:
            return 1.0
        ivp = 1.0 / np.diag(self.covariance_matrix)
        ivp /= ivp.sum()
        return ivp

    def get_cluster_var(self, cluster_idx):
        covariance_matrix_ = self.covariance_matrix[np.ix_(cluster_idx, cluster_idx)]
        w_ = self.get_ivp()
        cluster_var = np.dot(w_.T, np.dot(covariance_matrix_, w_))
        return cluster_var

    def bisect_clusters(self, clusters):
        clusters_new = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            n = len(cl) // 2
            clusters_new.extend([cl[:n], cl[n:]])
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

                alpha = min(np.sum(w_max[cl0]) / w[cl0[0]],
                            max(np.sum(w_min[cl0]) / w[cl0[0]], alpha))

                alpha = 1 - min(np.sum(w_max[cl1]) / w[cl1[0]],
                                max(np.sum(w_min[cl1]) / w[cl1[0]], 1 - alpha))

                w[cl0] *= alpha
                w[cl1] *= (1 - alpha)
            clusters = self.bisect_clusters(clusters)

        return w

    def hrp(self):
        if self.covariance_matrix is None and self.asset_returns is None:
            raise ValueError("Invalid input. Please provide either a covariance matrix or asset returns.")

        rho = np.corrcoef(self.covariance_matrix)
        distance = squareform(np.sqrt((1 - rho) / 2))

        if self.method == 'divisive':
            # Use k-means for divisive clustering
            def kmeans_bisect(data, k=2):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
                return [data[kmeans.labels_ == i] for i in range(k)]

            hcluster = kmeans_bisect(distance)
            sorted_idx = np.argsort(hcluster[0].sum(axis=0))  # This is a simplification; adjust as needed.
        else:
            link = linkage(distance, method=self.method)
            sorted_idx = leaves_list(link)

        if not isinstance(self.w_min, np.ndarray) or len(self.w_min) == 1:
            self.w_min = np.full(self.covariance_matrix.shape[0], self.w_min)
        if not isinstance(self.w_max, np.ndarray) or len(self.w_max) == 1:
            self.w_max = np.full(self.covariance_matrix.shape[0], self.w_max)
        if np.any(self.w_min > 1):
            raise ValueError("Invalid minimum weights.")

        w = self.get_rec_bipart(sorted_idx, self.w_min, self.w_max, self.lam)

        return {'weights': w, 'tree': dendrogram(link) if self.method != 'divisive' else None}

    def optimize(self):
        if self.asset_returns is None or len(self.asset_returns.columns) == 0:
            print("----------> Error: No assets available for optimization.")
            return None

        hrp_result = self.hrp()
        if hrp_result is None:
            print(f"----------> Optimization with Hierarchical Risk Parity did not converge.")
            return self.results

        weights = hrp_result['weights']
        selected_assets = self.asset_returns.columns[weights > 0]
        weights = weights[weights > 0]
        profit_percentage = calculate_portfolio_profit(self.asset_returns, selected_assets, weights)
        weights_dict = {selected_assets[i]: weights[i] * 100 for i in range(len(selected_assets))}
        self.results.append({
            'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': 'Hierarchical Risk Parity',
            'Weights': weights_dict,
            'Profit Percentage': profit_percentage
        })
        print(f"----------> Optimization with Hierarchical Risk Parity completed.")
        return self.results
