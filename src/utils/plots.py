"""File with all function used to plot the results"""

import networkx as nx
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt


def plot_tsp(p, x_coord, tour, title="default"):
    """
    Helper function to plot TSP tours.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        tour: Predicted tour
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    """
    W_val = squareform(pdist(x_coord, metric='euclidean'))
    G = nx.from_numpy_matrix(W_val)
    
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    
    tour_pairs = []
    for idx in range(len(tour)-1):
        tour_pairs.append((tour[idx], tour[idx+1]))
    tour_pairs.append((tour[idx+1], tour[0]))
    
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=30)
    nx.draw_networkx_edges(G, pos, edgelist=tour_pairs, edge_color='black', alpha=1, width=1)
    p.set_title(title)
    return p

def plot_tsp_heatmap(p, x_coord, W_pred, threshold=0.1, title="default"):
    """
    Helper function to plot predicted TSP tours with edge strength denoting confidence of prediction.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W_pred: Edge predictions matrix
        threshold: Threshold above which edge predicion probabilities are plotted
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    """
    W_val = squareform(pdist(x_coord, metric='euclidean'))
    G = nx.from_numpy_matrix(W_val)
    
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))    
    
    edge_pairs = []
    edge_color = []
    for r in range(len(W_pred)):
        for c in range(len(W_pred)):
            if W_pred[r][c] >= threshold:
                edge_pairs.append((r, c))
                edge_color.append(W_pred[r][c])
    
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=30)
    nx.draw_networkx_edges(G, pos, edgelist=edge_pairs, edge_color=edge_color, edge_cmap=plt.cm.Reds, width=1)
    p.set_title(title)
    return p