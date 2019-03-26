"""
@author: linlin
@references: Pierre Mahé and Jean-Philippe Vert. Graph kernels based on tree patterns for molecules. Machine learning, 75(1):3–35, 2009.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time

import networkx as nx
import numpy as np

from collections import Counter
from tqdm import tqdm
tqdm.monitor_interval = 0

from pygraph.utils.utils import untotterTransformation


def treepatternkernel(*args,
                      node_label='atom',
                      edge_label='bond_type',
                      labeled=True,
                      kernel_type='untiln',
                      lmda=1,
                      h=1,
                      remove_totters=True):
    """Calculate tree pattern graph kernels between graphs.
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
    labeled : boolean
        Whether the graphs are labeled. The default is True.
    kernel_type : string
        Type of tree pattern kernel, could be 'untiln', 'size' or 'branching'.
    lmda : float
        Weight to decide whether linear patterns or trees pattern of increasing complexity are favored.
    h : integer
        The upper bound of the height of tree patterns.
    remove_totters : boolean
        whether to remove totters. The default value is True.

    Return
    ------
    Kmatrix: Numpy matrix
        Kernel matrix, each element of which is the tree pattern graph kernel between 2 praphs.
    """
    if h < 1:
        raise Exception('h > 0 is requested.')
    kernel_type = kernel_type.lower()
    # arrange all graphs in a list
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    h = int(h)

    start_time = time.time()

    if remove_totters:
        Gn = [untotterTransformation(G, node_label, edge_label) for G in Gn]

    pbar = tqdm(
        total=(1 + len(Gn)) * len(Gn) / 2,
        desc='calculate kernels',
        file=sys.stdout)
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _treepatternkernel_do(Gn[i], Gn[j], node_label,
                                                  edge_label, labeled,
                                                  kernel_type, lmda, h)
            Kmatrix[j][i] = Kmatrix[i][j]
            pbar.update(1)

    run_time = time.time() - start_time
    print(
        "\n --- kernel matrix of tree pattern kernel of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def _treepatternkernel_do(G1, G2, node_label, edge_label, labeled, kernel_type,
                          lmda, h):
    """Calculate tree pattern graph kernels between 2 graphs.

    Parameters
    ----------
    paths1, paths2 : list
        List of paths in 2 graphs, where for unlabeled graphs, each path is represented by a list of nodes; while for labeled graphs, each path is represented by a string consists of labels of nodes and edges on that path.
    k_func : function
        A kernel function used using different notions of fingerprint similarity.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    labeled : boolean
        Whether the graphs are labeled. The default is True.
    kernel_type : string
        Type of tree pattern kernel, could be 'untiln', 'size' or 'branching'.
    lmda : float
        Weight to decide whether linear patterns or trees pattern of increasing complexity are favored.
    h : integer
        The upper bound of the height of tree patterns.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """

    def matchingset(n1, n2):
        """Get neiborhood matching set of two nodes in two graphs.
        """

        def mset_com(allpairs, length):
            """Find all sets R of pairs by combination.
            """
            if length == 1:
                mset = [[pair] for pair in allpairs]
                return mset, mset
            else:
                mset, mset_l = mset_com(allpairs, length - 1)
                mset_tmp = []
                for pairset in mset_l:  # for each pair set of length l-1
                    nodeset1 = [pair[0] for pair in pairset
                                ]  # nodes already in the set
                    nodeset2 = [pair[1] for pair in pairset]
                    for pair in allpairs:
                        if (pair[0] not in nodeset1) and (
                                pair[1] not in nodeset2
                        ):  # nodes in R should be unique
                            mset_tmp.append(
                                pairset + [pair]
                            )  # add this pair to the pair set of length l-1, constructing a new set of length l
                            nodeset1.append(pair[0])
                            nodeset2.append(pair[1])

                mset.extend(mset_tmp)

                return mset, mset_tmp

        allpairs = [
        ]  # all pairs those have the same node labels and edge labels
        for neighbor1 in G1[n1]:
            for neighbor2 in G2[n2]:
                if G1.node[neighbor1][node_label] == G2.node[neighbor2][node_label] \
                   and G1[n1][neighbor1][edge_label] == G2[n2][neighbor2][edge_label]:
                    allpairs.append([neighbor1, neighbor2])

        if allpairs != []:
            mset, _ = mset_com(allpairs, len(allpairs))
        else:
            mset = []

        return mset

    def kernel_h(h):
        """Calculate kernel of h-th iteration.
        """

        if kernel_type == 'untiln':
            all_kh = { str(n1) + '.' + str(n2) : (G1.node[n1][node_label] == G2.node[n2][node_label]) \
                for n1 in G1.nodes() for n2 in G2.nodes() } # kernels between all pair of nodes with h = 1 ]
            all_kh_tmp = all_kh.copy()
            for i in range(2, h + 1):
                for n1 in G1.nodes():
                    for n2 in G2.nodes():
                        kh = 0
                        mset = all_msets[str(n1) + '.' + str(n2)]
                        for R in mset:
                            kh_tmp = 1
                            for pair in R:
                                kh_tmp *= lmda * all_kh[str(pair[0])
                                                        + '.' + str(pair[1])]
                            kh += 1 / lmda * kh_tmp
                        kh = (G1.node[n1][node_label] == G2.node[n2][
                            node_label]) * (1 + kh)
                        all_kh_tmp[str(n1) + '.' + str(n2)] = kh
                all_kh = all_kh_tmp.copy()

        elif kernel_type == 'size':
            all_kh = { str(n1) + '.' + str(n2) : lmda * (G1.node[n1][node_label] == G2.node[n2][node_label]) \
                for n1 in G1.nodes() for n2 in G2.nodes() } # kernels between all pair of nodes with h = 1 ]
            all_kh_tmp = all_kh.copy()
            for i in range(2, h + 1):
                for n1 in G1.nodes():
                    for n2 in G2.nodes():
                        kh = 0
                        mset = all_msets[str(n1) + '.' + str(n2)]
                        for R in mset:
                            kh_tmp = 1
                            for pair in R:
                                kh_tmp *= lmda * all_kh[str(pair[0])
                                                        + '.' + str(pair[1])]
                            kh += kh_tmp
                        kh *= lmda * (
                            G1.node[n1][node_label] == G2.node[n2][node_label])
                        all_kh_tmp[str(n1) + '.' + str(n2)] = kh
                all_kh = all_kh_tmp.copy()

        elif kernel_type == 'branching':
            all_kh = { str(n1) + '.' + str(n2) : (G1.node[n1][node_label] == G2.node[n2][node_label]) \
                for n1 in G1.nodes() for n2 in G2.nodes() } # kernels between all pair of nodes with h = 1 ]
            all_kh_tmp = all_kh.copy()
            for i in range(2, h + 1):
                for n1 in G1.nodes():
                    for n2 in G2.nodes():
                        kh = 0
                        mset = all_msets[str(n1) + '.' + str(n2)]
                        for R in mset:
                            kh_tmp = 1
                            for pair in R:
                                kh_tmp *= lmda * all_kh[str(pair[0])
                                                        + '.' + str(pair[1])]
                            kh += 1 / lmda * kh_tmp
                        kh *= (
                            G1.node[n1][node_label] == G2.node[n2][node_label])
                        all_kh_tmp[str(n1) + '.' + str(n2)] = kh
                all_kh = all_kh_tmp.copy()

        return all_kh

    # calculate matching sets for every pair of nodes at first to avoid calculating in every iteration.
    all_msets = ({ str(node1) + '.' + str(node2) : matchingset(node1, node2) for node1 in G1.nodes() \
        for node2 in G2.nodes() } if h > 1 else {})

    all_kh = kernel_h(h)
    kernel = sum(all_kh.values())

    if kernel_type == 'size':
        kernel = kernel / (lmda**h)

    return kernel
