"""
@author: linlin
@references: S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M Borgwardt. Graph kernels. Journal of Machine Learning Research, 11(Apr):1201–1242, 2010.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time

# from collections import Counter

import networkx as nx
import numpy as np


def randomwalkkernel(*args, node_label='atom', edge_label='bond_type', labeled=True, n=10, method=''):
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
    labeled : boolean
        Whether the graphs are labeled. The default is True.
    n : integer
        Longest length of walks.
    method : string
        Method used to compute the random walk kernel. Available methods are 'sylvester', 'conjugate', 'fp', 'spectral' and 'kron'.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the path kernel up to d between 2 praphs.
    """
    method = method.lower()
    Gn = args[0] if len(args) == 1 else [args[0], args[1]] # arrange all graphs in a list
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    n = int(n)

    start_time = time.time()

    # get all paths of all graphs before calculating kernels to save time, but this may cost a lot of memory for large dataset.
    all_walks = [ find_all_walks_until_length(Gn[i], n, node_label = node_label, edge_label = edge_label, labeled = labeled) for i in range(0, len(Gn)) ]

    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _randomwalkkernel_do(all_walks[i], all_walks[j], node_label = node_label, edge_label = edge_label, labeled = labeled)
            Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print("\n --- kernel matrix of walk kernel up to %d of size %d built in %s seconds ---" % (n, len(Gn), run_time))

    return Kmatrix, run_time


def _randomwalkkernel_do(walks1, walks2, node_label = 'atom', edge_label = 'bond_type', labeled = True, method=''):
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

    if method == 'sylvester':
        import warnings
        warnings.warn('The Sylvester equation (rather than generalized Sylvester equation) is used; only walks of length 1 is considered.')
        from control import dlyap
        dpg = nx.tensor_product(G1, G2) # direct product graph
        X = dlyap(A, Q, C)
        pass

    else:
        raise Exception('No computation method specified.')

    return kernel

