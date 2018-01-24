import sys
import pathlib
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.deltaKernel import deltakernel

def pathkernel(*args, node_label = 'atom', edge_label = 'bond_type'):
    """Calculate mean average path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.

    Return
    ------
    Kmatrix/kernel : Numpy matrix/float
        Kernel matrix, each element of which is the path kernel between 2 praphs. / Path kernel between 2 graphs.

    References
    ----------
    [1] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).
    """
    some_graph = args[0][0] if len(args) == 1 else args[0] # only edge attributes of type int or float can be used as edge weight to calculate the shortest paths.
    some_weight = list(nx.get_edge_attributes(some_graph, edge_label).values())[0]
    weight = edge_label if isinstance(some_weight, float) or isinstance(some_weight, int) else None

    if len(args) == 1: # for a list of graphs
        Gn = args[0]
        Kmatrix = np.zeros((len(Gn), len(Gn)))

        start_time = time.time()

        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _pathkernel_do(Gn[i], Gn[j], node_label, edge_label, weight = weight)
                Kmatrix[j][i] = Kmatrix[i][j]

        run_time = time.time() - start_time
        print("\n --- mean average path kernel matrix of size %d built in %s seconds ---" % (len(Gn), run_time))

        return Kmatrix, run_time

    else: # for only 2 graphs
        start_time = time.time()

        kernel = _pathkernel_do(args[0], args[1], node_label, edge_label, weight = weight)

        run_time = time.time() - start_time
        print("\n --- mean average path kernel built in %s seconds ---" % (run_time))

        return kernel, run_time


def _pathkernel_do(G1, G2, node_label = 'atom', edge_label = 'bond_type', weight = None):
    """Calculate mean average path kernel between 2 graphs.

    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    weight : string/None
        edge attribute used as weight to calculate the shortest path. The default edge label is None.

    Return
    ------
    kernel : float
        Path Kernel between 2 graphs.
    """
    # calculate shortest paths for both graphs
    sp1 = []
    num_nodes = G1.number_of_nodes()
    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
                sp1.append(nx.shortest_path(G1, node1, node2, weight = weight))

    sp2 = []
    num_nodes = G2.number_of_nodes()
    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
                sp2.append(nx.shortest_path(G2, node1, node2, weight = weight))

    # calculate kernel
    kernel = 0
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                kernel_path = deltakernel(G1.node[path1[0]][node_label] == G2.node[path2[0]][node_label])
                if kernel_path:
                    for i in range(1, len(path1)):
                         # kernel = 1 if all corresponding nodes and edges in the 2 paths have same labels, otherwise 0
                        kernel_path *= deltakernel(G1[path1[i - 1]][path1[i]][edge_label] == G2[path2[i - 1]][path2[i]][edge_label]) * deltakernel(G1.node[path1[i]][node_label] == G2.node[path2[i]][node_label])
                    kernel += kernel_path # add up kernels of all paths

    kernel = kernel / (len(sp1) * len(sp2)) # calculate mean average

    return kernel
