import networkx as nx
import numpy as np


def getSPLengths(G1):
    sp = nx.shortest_path(G1)
    distances = np.zeros((G1.number_of_nodes(), G1.number_of_nodes()))
    for i in sp.keys():
        for j in sp[i].keys():
            distances[i, j] = len(sp[i][j])-1
    return distances

def getSPGraph(G, edge_weight = 'bond_type'):
    """Transform graph G to its corresponding shortest-paths graph.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.
        
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
        
    Notes
    ------
    For an input graph G, its corresponding shortest-paths graph S contains the same set of nodes as G, while there exists an edge between all nodes in S which are connected by a walk in G. Every edge in S between two nodes is labeled by the shortest distance between these two nodes.
    
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    return floydTransformation(G, edge_weight = edge_weight)
            
def floydTransformation(G, edge_weight = 'bond_type'):
    """Transform graph G to its corresponding shortest-paths graph using Floyd-transformation.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.
        
    Return
    ------
    S : NetworkX graph
        The shortest-paths graph corresponding to G.
        
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    spMatrix = nx.floyd_warshall_numpy(G, weight = edge_weight)
    S = nx.Graph()
    S.add_nodes_from(G.nodes(data=True))
    for i in range(0, G.number_of_nodes()):
        for j in range(0, G.number_of_nodes()):
            S.add_edge(i, j, cost = spMatrix[i, j])
    return S
