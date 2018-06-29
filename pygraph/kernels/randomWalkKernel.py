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
                     edge_weight=None,
                     h=10,
                     p=None,
                     q=None,
                     weight=None,
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
    h : integer
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

    eweight = None
    if edge_weight == None:
        print('\n None edge weight specified. Set all weight to 1.\n')
    else:
        try:
            some_weight = list(
                nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
            if isinstance(some_weight, float) or isinstance(some_weight, int):
                eweight = edge_weight
            else:
                print(
                    '\n Edge weight with name %s is not float or integer. Set all weight to 1.\n'
                    % edge_weight)
        except:
            print(
                '\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n'
                % edge_weight)

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

    if compute_method == 'sylvester':
        import warnings
        warnings.warn(
            'The Sylvester equation (rather than generalized Sylvester equation) is used; edge label number has to smaller than 3.'
        )
        Kmatrix = _randomwalkkernel_sylvester(Gn, weight, p, q, node_label,
                                              edge_label, eweight)

    elif compute_method == 'conjugate':
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _randomwalkkernel_conjugate(
                    Gn[i], Gn[j], node_label, edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    elif compute_method == 'fp':
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _randomwalkkernel_fp(Gn[i], Gn[j], node_label,
                                                     edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    elif compute_method == 'spectral':
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _randomwalkkernel_spectral(
                    Gn[i], Gn[j], node_label, edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)
    elif compute_method == 'kron':
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _randomwalkkernel_kron(Gn[i], Gn[j],
                                                       node_label, edge_label)
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)
    else:
        raise Exception(
            'compute method name incorrect. Available methods: "sylvester", "conjugate", "fp", "spectral" and "kron".'
        )

    # for i in range(0, len(Gn)):
    #     for j in range(i, len(Gn)):
    #         Kmatrix[i][j] = _randomwalkkernel_do(
    #             all_walks[i],
    #             all_walks[j],
    #             node_label=node_label,
    #             edge_label=edge_label,
    #             labeled=labeled)
    #         Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print(
        "\n --- kernel matrix of random walk kernel of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def _randomwalkkernel_sylvester(Gn, lmda, p, q, node_label, edge_label,
                                eweight):
    """Calculate walk graph kernels up to n between 2 graphs using Sylvester method.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.

    Return
    ------
    kernel : float
        Kernel between 2 graphs.
    """
    from control import dlyap
    Kmatrix = np.zeros((len(Gn), len(Gn)))

    if q == None:
        # don't normalize adjacency matrices if q is a uniform vector.
        A_list = [
            nx.adjacency_matrix(G, eweight).todense() for G in tqdm(
                Gn, desc='compute adjacency matrices', file=sys.stdout)
        ]
        if p == None:
            pbar = tqdm(
                total=(1 + len(Gn)) * len(Gn) / 2,
                desc='calculating kernels',
                file=sys.stdout)
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    A = lmda * A_list[j]
                    Q = A_list[i]
                    # use uniform distribution if there is no prior knowledge.
                    nb_pd = len(A_list[i]) * len(A_list[j])
                    pd_uni = 1 / nb_pd
                    C = np.full((len(A_list[j]), len(A_list[i])), pd_uni)
                    try:
                        X = dlyap(A, Q, C)
                        X = np.reshape(X, (-1, 1), order='F')
                        # use uniform distribution if there is no prior knowledge.
                        q_direct = np.full((1, nb_pd), pd_uni)
                        Kmatrix[i][j] = np.dot(q_direct, X)
                    except TypeError:
                        # print('sth wrong.')
                        Kmatrix[i][j] = np.nan

                    Kmatrix[j][i] = Kmatrix[i][j]
                    pbar.update(1)
    # A_list = []
    # for G in tqdm(Gn, desc='compute adjacency matrices', file=sys.stdout):
    #     A_tilde = nx.adjacency_matrix(G, weight=None).todense()
    #     # normalized adjacency matrices
    #     #          A_list.append(A_tilde / A_tilde.sum(axis=0))
    #     A_list.append(A_tilde)

    return Kmatrix


def _randomwalkkernel_conjugate(G1, G2, node_label, edge_label):
    """Calculate walk graph kernels up to n between 2 graphs using conjugate method.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.

    Return
    ------
    kernel : float
        Kernel between 2 graphs.
    """

    dpg = nx.tensor_product(G1, G2)  # direct product graph
    import matplotlib.pyplot as plt
    nx.draw_networkx(G1)
    plt.show()
    nx.draw_networkx(G2)
    plt.show()
    nx.draw_networkx(dpg)
    plt.show()
    X = dlyap(A, Q, C)

    return kernel


def _randomwalkkernel_fp(G1, G2, node_label, edge_label):
    """Calculate walk graph kernels up to n between 2 graphs using Fixed-Point method.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.

    Return
    ------
    kernel : float
        Kernel between 2 graphs.
    """

    dpg = nx.tensor_product(G1, G2)  # direct product graph
    X = dlyap(A, Q, C)

    return kernel


def _randomwalkkernel_spectral(G1, G2, node_label, edge_label):
    """Calculate walk graph kernels up to n between 2 graphs using spectral decomposition method.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.

    Return
    ------
    kernel : float
        Kernel between 2 graphs.
    """

    dpg = nx.tensor_product(G1, G2)  # direct product graph
    X = dlyap(A, Q, C)

    return kernel


def _randomwalkkernel_kron(G1, G2, node_label, edge_label):
    """Calculate walk graph kernels up to n between 2 graphs using nearest Kronecker product approximation method.

    Parameters
    ----------
    G1, G2 : NetworkX graph
        Graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.

    Return
    ------
    kernel : float
        Kernel between 2 graphs.
    """

    dpg = nx.tensor_product(G1, G2)  # direct product graph
    X = dlyap(A, Q, C)

    return kernel
