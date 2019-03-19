"""
@author: linlin
@references: Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time
import itertools
from tqdm import tqdm

import networkx as nx
import numpy as np

from pygraph.kernels.deltaKernel import deltakernel
from pygraph.utils.graphdataset import get_dataset_attributes


def pathkernel(*args, node_label='atom', edge_label='bond_type'):
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
    """
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    ds_attrs = get_dataset_attributes(
        Gn,
        attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
        node_label=node_label,
        edge_label=edge_label)
    try:
        some_weight = list(nx.get_edge_attributes(Gn[0],
                                                  edge_label).values())[0]
        weight = edge_label if isinstance(some_weight, float) or isinstance(
            some_weight, int) else None
    except:
        weight = None

    start_time = time.time()

    splist = [
        get_shortest_paths(Gn[i], weight) for i in tqdm(
            range(0, len(Gn)), desc='getting shortest paths', file=sys.stdout)
    ]

    pbar = tqdm(
        total=((len(Gn) + 1) * len(Gn) / 2),
        desc='calculating kernels',
        file=sys.stdout)
    if ds_attrs['node_labeled']:
        if ds_attrs['edge_labeled']:
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    Kmatrix[i][j] = _pathkernel_do_l(Gn[i], Gn[j], splist[i],
                                                     splist[j], node_label,
                                                     edge_label)
                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)
        else:
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    Kmatrix[i][j] = _pathkernel_do_nl(Gn[i], Gn[j], splist[i],
                                                      splist[j], node_label)
                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)

    else:
        if ds_attrs['edge_labeled']:
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    Kmatrix[i][j] = _pathkernel_do_el(Gn[i], Gn[j], splist[i],
                                                      splist[j], edge_label)
                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)
        else:
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    Kmatrix[i][j] = _pathkernel_do_unl(Gn[i], Gn[j], splist[i],
                                                       splist[j])
                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)

    run_time = time.time() - start_time
    print(
        "\n --- mean average path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def _pathkernel_do_l(G1, G2, sp1, sp2, node_label, edge_label):
    """Calculate mean average path kernel between 2 fully-labeled graphs.

    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    sp1, sp2 : list of list
        List of shortest paths of 2 graphs, where each path is represented by a list of nodes.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.

    Return
    ------
    kernel : float
        Path Kernel between 2 graphs.
    """
    # calculate kernel
    kernel = 0
    #     if len(sp1) == 0 or len(sp2) == 0:
    #         return 0 # @todo: should it be zero?
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                kernel_path = (G1.node[path1[0]][node_label] == G2.node[path2[
                    0]][node_label])
                if kernel_path:
                    for i in range(1, len(path1)):
                        # kernel = 1 if all corresponding nodes and edges in the 2 paths have same labels, otherwise 0
                        if G1[path1[i - 1]][path1[i]][edge_label] != G2[path2[i - 1]][path2[i]][edge_label] or G1.node[path1[i]][node_label] != G2.node[path2[i]][node_label]:
                            kernel_path = 0
                            break
                    kernel += kernel_path  # add up kernels of all paths

    kernel = kernel / (len(sp1) * len(sp2))  # calculate mean average

    return kernel


def _pathkernel_do_nl(G1, G2, sp1, sp2, node_label):
    """Calculate mean average path kernel between 2 node-labeled graphs.
    """
    # calculate kernel
    kernel = 0
    #     if len(sp1) == 0 or len(sp2) == 0:
    #         return 0 # @todo: should it be zero?
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                kernel_path = 1
                for i in range(0, len(path1)):
                    # kernel = 1 if all corresponding nodes in the 2 paths have same labels, otherwise 0
                    if G1.node[path1[i]][node_label] != G2.node[path2[i]][node_label]:
                        kernel_path = 0
                        break
                kernel += kernel_path

    kernel = kernel / (len(sp1) * len(sp2))  # calculate mean average

    return kernel


def _pathkernel_do_el(G1, G2, sp1, sp2, edge_label):
    """Calculate mean average path kernel between 2 edge-labeled graphs.
    """
    # calculate kernel
    kernel = 0
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                if len(path1) == 0:
                    kernel += 1
                else:
                    kernel_path = 1
                    for i in range(0, len(path1) - 1):
                        # kernel = 1 if all corresponding edges in the 2 paths have same labels, otherwise 0
                        if G1[path1[i]][path1[i + 1]][edge_label] != G2[path2[
                                i]][path2[i + 1]][edge_label]:
                            kernel_path = 0
                            break
                    kernel += kernel_path

    kernel = kernel / (len(sp1) * len(sp2))  # calculate mean average

    return kernel


def _pathkernel_do_unl(G1, G2, sp1, sp2):
    """Calculate mean average path kernel between 2 unlabeled graphs.
    """
    # calculate kernel
    kernel = 0
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                kernel += 1

    kernel = kernel / (len(sp1) * len(sp2))  # calculate mean average

    return kernel


def get_shortest_paths(G, weight):
    """Get all shortest paths of a graph.

    Parameters
    ----------
    G : NetworkX graphs
        The graphs whose paths are calculated.
    weight : string/None
        edge attribute used as weight to calculate the shortest path.

    Return
    ------
    sp : list of list
        List of shortest paths of the graph, where each path is represented by a list of nodes.
    """
    sp = []
    for n1, n2 in itertools.combinations(G.nodes(), 2):
        try:
            sp.append(nx.shortest_path(G, n1, n2, weight=weight))
        except nx.NetworkXNoPath:  # nodes not connected
            sp.append([])
    # add single nodes as length 0 paths.
    sp += [[n] for n in G.nodes()]
    return sp