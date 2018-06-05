"""
@author: linlin
@references:
    [1] Thomas Gärtner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. Learning Theory and Kernel Machines, pages 129–143, 2003.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time
from tqdm import tqdm
from collections import Counter
from itertools import product

import networkx as nx
import numpy as np

from pygraph.utils.utils import direct_product
from pygraph.utils.graphdataset import get_dataset_attributes


def commonwalkkernel(*args,
                     node_label='atom',
                     edge_label='bond_type',
                     n=None,
                     weight=1,
                     compute_method='exp'):
    """Calculate common walk graph kernels up to depth d between graphs.
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
    n : integer
        Longest length of walks.
    weight: integer
        Weight coefficient of different lengths of walks.
    compute_method : string
        Method used to compute walk kernel. The Following choices are available:
        'direct' : direct product graph method, as shown in reference [1]. The time complexity is O(n^6) for unlabeled graphs with n vertices.
        'brute' : brute force, simply search for all walks and compare them.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the path kernel up to d between 2 graphs.
    """
    compute_method = compute_method.lower()
    # arrange all graphs in a list
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    ds_attrs = get_dataset_attributes(
        Gn,
        attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
        node_label=node_label,
        edge_label=edge_label)
    if not ds_attrs['node_labeled']:
        for G in Gn:
            nx.set_node_attributes(G, '0', 'atom')
    if not ds_attrs['edge_labeled']:
        for G in Gn:
            nx.set_edge_attributes(G, '0', 'bond_type')

    start_time = time.time()

    # direct product graph method - exponential
    if compute_method == 'exp':
        pbar = tqdm(
            total=(1 + len(Gn)) * len(Gn) / 2,
            desc='calculating kernels',
            file=sys.stdout)
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _untilnwalkkernel_exp(Gn[i], Gn[j], node_label,
                                                      edge_label, weight)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    # direct product graph method - geometric
    if compute_method == 'geo':
        pbar = tqdm(
            total=(1 + len(Gn)) * len(Gn) / 2,
            desc='calculating kernels',
            file=sys.stdout)
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _untilnwalkkernel_geo(Gn[i], Gn[j], node_label,
                                                      edge_label, weight)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    # search all paths use brute force.
    elif compute_method == 'brute':
        n = int(n)
        # get all paths of all graphs before calculating kernels to save time, but this may cost a lot of memory for large dataset.
        all_walks = [
            find_all_walks_until_length(Gn[i], n, node_label, edge_label,
                                        labeled) for i in range(0, len(Gn))
        ]

        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _untilnwalkkernel_brute(
                    all_walks[i],
                    all_walks[j],
                    node_label=node_label,
                    edge_label=edge_label,
                    labeled=labeled)
                Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print(
        "\n --- kernel matrix of common walk kernel of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def _untilnwalkkernel_exp(G1, G2, node_label, edge_label, beta):
    """Calculate walk graph kernels up to n between 2 graphs using exponential series.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        Node attribute used as label.
    edge_label : string
        Edge attribute used as label.
    beta: integer
        Weight.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """

    # get tensor product / direct product
    gp = direct_product(G1, G2, node_label, edge_label)
    A = nx.adjacency_matrix(gp).todense()
    # print(A)

    # from matplotlib import pyplot as plt
    # nx.draw_networkx(G1)
    # plt.show()
    # nx.draw_networkx(G2)
    # plt.show()
    # nx.draw_networkx(gp)
    # plt.show()
    # print(G1.nodes(data=True))
    # print(G2.nodes(data=True))
    # print(gp.nodes(data=True))
    # print(gp.edges(data=True))

    ew, ev = np.linalg.eig(A)
    # print('ew: ', ew)
    # print(ev)
    # T = np.matrix(ev)
    # print('T: ', T)
    # T = ev.I
    D = np.zeros((len(ew), len(ew)))
    for i in range(len(ew)):
        D[i][i] = np.exp(beta * ew[i])
    # print('D: ', D)
    # print('hshs: ', T.I * D * T)

    # print(np.exp(-2))
    # print(D)
    # print(np.exp(weight * D))
    # print(ev)
    # print(np.linalg.inv(ev))
    exp_D = ev * D * ev.I
    # print(exp_D)
    # print(np.exp(weight * A))
    # print('-------')

    return np.sum(exp_D.diagonal())


def _untilnwalkkernel_geo(G1, G2, node_label, edge_label, gamma):
    """Calculate walk graph kernels up to n between 2 graphs using geometric series.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        Node attribute used as label.
    edge_label : string
        Edge attribute used as label.
    gamma: integer
        Weight.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """

    # get tensor product / direct product
    gp = direct_product(G1, G2, node_label, edge_label)
    A = nx.adjacency_matrix(gp).todense()
    # print(A)

    # from matplotlib import pyplot as plt
    # nx.draw_networkx(G1)
    # plt.show()
    # nx.draw_networkx(G2)
    # plt.show()
    # nx.draw_networkx(gp)
    # plt.show()
    # print(G1.nodes(data=True))
    # print(G2.nodes(data=True))
    # print(gp.nodes(data=True))
    # print(gp.edges(data=True))

    ew, ev = np.linalg.eig(A)
    # print('ew: ', ew)
    # print(ev)
    # T = np.matrix(ev)
    # print('T: ', T)
    # T = ev.I
    D = np.zeros((len(ew), len(ew)))
    for i in range(len(ew)):
        D[i][i] = np.exp(beta * ew[i])
    # print('D: ', D)
    # print('hshs: ', T.I * D * T)

    # print(np.exp(-2))
    # print(D)
    # print(np.exp(weight * D))
    # print(ev)
    # print(np.linalg.inv(ev))
    exp_D = ev * D * ev.I
    # print(exp_D)
    # print(np.exp(weight * A))
    # print('-------')

    return np.sum(exp_D.diagonal())


def _untilnwalkkernel_brute(walks1,
                            walks2,
                            node_label='atom',
                            edge_label='bond_type',
                            labeled=True):
    """Calculate walk graph kernels up to n between 2 graphs.

    Parameters
    ----------
    walks1, walks2 : list
        List of walks in 2 graphs, where for unlabeled graphs, each walk is represented by a list of nodes; while for labeled graphs, each walk is represented by a string consists of labels of nodes and edges on that walk.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    labeled : boolean
        Whether the graphs are labeled. The default is True.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """
    counts_walks1 = dict(Counter(walks1))
    counts_walks2 = dict(Counter(walks2))
    all_walks = list(set(walks1 + walks2))

    vector1 = [(counts_walks1[walk] if walk in walks1 else 0)
               for walk in all_walks]
    vector2 = [(counts_walks2[walk] if walk in walks2 else 0)
               for walk in all_walks]
    kernel = np.dot(vector1, vector2)

    return kernel


# this method find walks repetively, it could be faster.
def find_all_walks_until_length(G,
                                length,
                                node_label='atom',
                                edge_label='bond_type',
                                labeled=True):
    """Find all walks with a certain maximum length in a graph. A recursive depth first search is applied.

    Parameters
    ----------
    G : NetworkX graphs
        The graph in which walks are searched.
    length : integer
        The maximum length of walks.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    labeled : boolean
        Whether the graphs are labeled. The default is True.

    Return
    ------
    walk : list
        List of walks retrieved, where for unlabeled graphs, each walk is represented by a list of nodes; while for labeled graphs, each walk is represented by a string consists of labels of nodes and edges on that walk.
    """
    all_walks = []
    # @todo: in this way, the time complexity is close to N(d^n+d^(n+1)+...+1), which could be optimized to O(Nd^n)
    for i in range(0, length + 1):
        new_walks = find_all_walks(G, i)
        if new_walks == []:
            break
        all_walks.extend(new_walks)

    if labeled == True:  # convert paths to strings
        walk_strs = []
        for walk in all_walks:
            strlist = [
                G.node[node][node_label] +
                G[node][walk[walk.index(node) + 1]][edge_label]
                for node in walk[:-1]
            ]
            walk_strs.append(''.join(strlist) + G.node[walk[-1]][node_label])

        return walk_strs

    return all_walks


def find_walks(G, source_node, length):
    """Find all walks with a certain length those start from a source node. A recursive depth first search is applied.

    Parameters
    ----------
    G : NetworkX graphs
        The graph in which walks are searched.
    source_node : integer
        The number of the node from where all walks start.
    length : integer
        The length of walks.

    Return
    ------
    walk : list of list
        List of walks retrieved, where each walk is represented by a list of nodes.
    """
    return [[source_node]] if length == 0 else \
        [ [source_node] + walk for neighbor in G[source_node] \
        for walk in find_walks(G, neighbor, length - 1) ]


def find_all_walks(G, length):
    """Find all walks with a certain length in a graph. A recursive depth first search is applied.

    Parameters
    ----------
    G : NetworkX graphs
        The graph in which walks are searched.
    length : integer
        The length of walks.

    Return
    ------
    walk : list of list
        List of walks retrieved, where each walk is represented by a list of nodes.
    """
    all_walks = []
    for node in G:
        all_walks.extend(find_walks(G, node, length))

    ### The following process is not carried out according to the original article
    # all_paths_r = [ path[::-1] for path in all_paths ]

    # # For each path, two presentation are retrieved from its two extremities. Remove one of them.
    # for idx, path in enumerate(all_paths[:-1]):
    #     for path2 in all_paths_r[idx+1::]:
    #         if path == path2:
    #             all_paths[idx] = []
    #             break

    # return list(filter(lambda a: a != [], all_paths))
    return all_walks
