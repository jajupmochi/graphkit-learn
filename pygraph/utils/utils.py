import networkx as nx
import numpy as np

# from tqdm import tqdm


def getSPLengths(G1):
    sp = nx.shortest_path(G1)
    distances = np.zeros((G1.number_of_nodes(), G1.number_of_nodes()))
    for i in sp.keys():
        for j in sp[i].keys():
            distances[i, j] = len(sp[i][j]) - 1
    return distances


def getSPGraph(G, edge_weight='bond_type'):
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
    return floydTransformation(G, edge_weight=edge_weight)


def floydTransformation(G, edge_weight='bond_type'):
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
    spMatrix = nx.floyd_warshall_numpy(G, weight=edge_weight)
    S = nx.Graph()
    S.add_nodes_from(G.nodes(data=True))
    for i in range(0, G.number_of_nodes()):
        for j in range(i, G.number_of_nodes()):
            S.add_edge(i, j, cost=spMatrix[i, j])
    return S


def untotterTransformation(G, node_label, edge_label):
    """Transform graph G according to Mahé et al.'s work to filter out tottering patterns of marginalized kernel and tree pattern kernel.

    Parameters
    ----------
    G : NetworkX graph
        The graph to be tramsformed.
    node_label : string
        node attribute used as label. The default node label is 'atom'.
    edge_label : string
        edge attribute used as label. The default edge label is 'bond_type'.

    Return
    ------
    gt : NetworkX graph
        The shortest-paths graph corresponding to G.

    References
    ----------
    [1] Pierre Mahé, Nobuhisa Ueda, Tatsuya Akutsu, Jean-Luc Perret, and Jean-Philippe Vert. Extensions of marginalized graph kernels. In Proceedings of the twenty-first international conference on Machine learning, page 70. ACM, 2004.
    """
    # arrange all graphs in a list
    G = G.to_directed()
    gt = nx.Graph()
    gt.graph = G.graph
    gt.add_nodes_from(G.nodes(data=True))
    for edge in G.edges():
        gt.add_node(edge)
        gt.node[edge].update({node_label: G.node[edge[1]][node_label]})
        gt.add_edge(edge[0], edge)
        gt.edges[edge[0], edge].update({
            edge_label:
            G[edge[0]][edge[1]][edge_label]
        })
        for neighbor in G[edge[1]]:
            if neighbor != edge[0]:
                gt.add_edge(edge, (edge[1], neighbor))
                gt.edges[edge, (edge[1], neighbor)].update({
                    edge_label:
                    G[edge[1]][neighbor][edge_label]
                })
    # nx.draw_networkx(gt)
    # plt.show()

    # relabel nodes using consecutive integers for convenience of kernel calculation.
    gt = nx.convert_node_labels_to_integers(
        gt, first_label=0, label_attribute='label_orignal')
    return gt
