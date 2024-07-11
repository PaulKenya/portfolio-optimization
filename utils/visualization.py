import matplotlib.pyplot as plt
import networkx as nx


def visualize_portfolio(weights):
    assets = list(weights.keys())
    allocations = list(weights.values())

    plt.figure(figsize=(10, 6))
    plt.bar(assets, allocations)
    plt.xlabel('Assets')
    plt.ylabel('Allocation')
    plt.title('Portfolio Allocation')
    plt.show()


def visualize_graph(graph: nx.Graph, title: str):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')
    plt.title(title)
    plt.show()
