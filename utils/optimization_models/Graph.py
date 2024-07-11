import networkx as nx
import numpy as np
import pandas as pd


class Graph:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.correlation_matrix = data.corr().fillna(0)
        self._mst = None
        self._tmfg = None

    def mst(self) -> nx.Graph:
        if self._mst is not None:
            print("----------> Retrieving stored MST Graph")
            return self._mst

        print("----------> Starting MST Graph generation")
        graph = nx.Graph()
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                weight = 1 - self.correlation_matrix.iloc[i, j]
                graph.add_edge(self.correlation_matrix.columns[i], self.correlation_matrix.columns[j], weight=weight)

        self._mst = nx.minimum_spanning_tree(graph, weight='weight')
        print("----------> Completed MST Graph generation")
        return self._mst

    def tmfg(self, k: int = 3) -> nx.Graph:
        if self._tmfg is not None:
            print("----------> Retrieving stored TMFG Graph")
            return self._tmfg

        print("----------> Starting TMFG Graph generation")
        graph = nx.Graph()
        distance_matrix = np.sqrt(0.5 * (1 - self.correlation_matrix))

        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                graph.add_edge(self.correlation_matrix.index[i], self.correlation_matrix.columns[j], weight=distance_matrix.iloc[i, j])

        graph_tmfg = nx.Graph()
        for node in graph.nodes():
            neighbors = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[:k]
            for neighbor in neighbors:
                graph_tmfg.add_edge(node, neighbor[0], weight=neighbor[1]['weight'])

        self._tmfg = graph_tmfg
        print("----------> Completed TMFG Graph generation")
        return self._tmfg
