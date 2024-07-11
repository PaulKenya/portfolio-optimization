import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


class HierarchicalRiskParity:
    def __init__(self, returns):
        self.returns = returns
        self.correlation_matrix = None
        self.linkage_matrix = None
        self.weights = None

    def calculate_correlation_matrix(self):
        self.correlation_matrix = self.returns.corr()

    def perform_clustering(self):
        self.linkage_matrix = linkage(self.correlation_matrix, method='single')

    def allocate_weights(self):
        inverse_variance = np.diag(np.linalg.inv(self.correlation_matrix))
        risk_contributions = inverse_variance / np.sum(inverse_variance)
        self.weights = risk_contributions / np.sum(risk_contributions)

    def plot_dendrogram(self):
        dendrogram(self.linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Asset')
        plt.ylabel('Distance')

        plt.show()
