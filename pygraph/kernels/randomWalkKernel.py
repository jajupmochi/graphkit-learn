"""
@author: linlin
@references: S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time
from tqdm import tqdm
# from collections import Counter

import networkx as nx
import numpy as np

from pygraph.utils.graphdataset import get_dataset_attributes


def randomwalkkernel(*args,
                     node_label='atom',
                     edge_label='bond_type',
                     h=10,
                     compute_method=''):
    """Calculate random walk graph kernels.
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
    method : string
        Method used to compute the random walk kernel. Available methods are 'sylvester', 'conjugate', 'fp', 'spectral' and 'kron'.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the path kernel up to d between 2 praphs.
    """
    compute_method = compute_method.lower()
    h = int(h)
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

    # # get all paths of all graphs before calculating kernels to save time, but this may cost a lot of memory for large dataset.
    # all_walks = [
    #     find_all_walks_until_length(
    #         Gn[i],
    #         n,
    #         node_label=node_label,
    #         edge_label=edge_label,
    #         labeled=labeled) for i in range(0, len(Gn))
    # ]

    pbar = tqdm(
        total=(1 + len(Gn)) * len(Gn) / 2,
        desc='calculating kernels',
        file=sys.stdout)
    if compute_method == 'sylvester':
        import warnings
        warnings.warn(
            'The Sylvester equation (rather than generalized Sylvester equation) is used; only walks of length 1 is considered.'
        )
        from control import dlyap
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _randomwalkkernel_sylvester(
                    all_walks[i],
                    all_walks[j],
                    node_label=node_label,
                    edge_label=edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    elif compute_method == 'conjugate':
        pass
    elif compute_method == 'fp':
        pass
    elif compute_method == 'spectral':
        pass
    elif compute_method == 'kron':
        pass
    else:
        raise Exception(
            'compute method name incorrect. Available methods: "sylvester", "conjugate", "fp", "spectral" and "kron".'
        )

    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _randomwalkkernel_do(
                all_walks[i],
                all_walks[j],
                node_label=node_label,
                edge_label=edge_label,
                labeled=labeled)
            Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print(
        "\n --- kernel matrix of walk kernel up to %d of size %d built in %s seconds ---"
        % (n, len(Gn), run_time))

    return Kmatrix, run_time


def _randomwalkkernel_sylvester(walks1,
                                walks2,
                                node_label='atom',
                                edge_label='bond_type'):
    """Calculate walk graph kernels up to n between 2 graphs using Sylvester method.

    Parameters
    ----------
    walks1, walks2 : list
        List of walks in 2 graphs, where for unlabeled graphs, each walk is represented by a list of nodes; while for labeled graphs, each walk is represented by a string consists of labels of nodes and edges on that walk.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """

    dpg = nx.tensor_product(G1, G2)  # direct product graph
    X = dlyap(A, Q, C)

    return kernel
