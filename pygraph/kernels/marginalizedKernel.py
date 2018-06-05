"""
@author: linlin
@references:
    [1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.
    [2] Pierre Mahé, Nobuhisa Ueda, Tatsuya Akutsu, Jean-Luc Perret, and Jean-Philippe Vert. Extensions of marginalized graph kernels. In Proceedings of the twenty-first international conference on Machine learning, page 70. ACM, 2004.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time
from tqdm import tqdm
tqdm.monitor_interval = 0

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from pygraph.kernels.deltaKernel import deltakernel
from pygraph.utils.utils import untotterTransformation
from pygraph.utils.graphdataset import get_dataset_attributes


def marginalizedkernel(*args,
                       node_label='atom',
                       edge_label='bond_type',
                       p_quit=0.5,
                       itr=20,
                       remove_totters=True):
    """Calculate marginalized graph kernels between graphs.

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
    p_quit : integer
        the termination probability in the random walks generating step
    itr : integer
        time of iterations to calculate R_inf
    remove_totters : boolean
        whether to remove totters. The default value is True.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the marginalized kernel between 2 praphs.
    """
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

    if remove_totters:
        Gn = [
            untotterTransformation(G, node_label, edge_label)
            for G in tqdm(Gn, desc='removing tottering', file=sys.stdout)
        ]

    pbar = tqdm(
        total=(1 + len(Gn)) * len(Gn) / 2,
        desc='calculating kernels',
        file=sys.stdout)
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _marginalizedkernel_do(Gn[i], Gn[j], node_label,
                                                   edge_label, p_quit, itr)
            Kmatrix[j][i] = Kmatrix[i][j]
            pbar.update(1)

    run_time = time.time() - start_time
    print(
        "\n --- marginalized kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def _marginalizedkernel_do(G1, G2, node_label, edge_label, p_quit, itr):
    """Calculate marginalized graph kernel between 2 graphs.

    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.
    p_quit : integer
        the termination probability in the random walks generating step.
    itr : integer
        time of iterations to calculate R_inf.

    Return
    ------
    kernel : float
        Marginalized Kernel between 2 graphs.
    """
    # init parameters
    kernel = 0
    num_nodes_G1 = nx.number_of_nodes(G1)
    num_nodes_G2 = nx.number_of_nodes(G2)
    p_init_G1 = 1 / num_nodes_G1  # the initial probability distribution in the random walks generating step (uniform distribution over |G|)
    p_init_G2 = 1 / num_nodes_G2

    q = p_quit * p_quit
    r1 = q

    # initial R_inf
    # matrix to save all the R_inf for all pairs of nodes
    R_inf = np.zeros([num_nodes_G1, num_nodes_G2])

    # calculate R_inf with a simple interative method
    for i in range(1, itr):
        R_inf_new = np.zeros([num_nodes_G1, num_nodes_G2])
        R_inf_new.fill(r1)

        # calculate R_inf for each pair of nodes
        for node1 in G1.nodes(data=True):
            neighbor_n1 = G1[node1[0]]
            # the transition probability distribution in the random walks generating step (uniform distribution over the vertices adjacent to the current vertex)
            p_trans_n1 = (1 - p_quit) / len(neighbor_n1)
            for node2 in G2.nodes(data=True):
                neighbor_n2 = G2[node2[0]]
                p_trans_n2 = (1 - p_quit) / len(neighbor_n2)

                for neighbor1 in neighbor_n1:
                    for neighbor2 in neighbor_n2:
                        t = p_trans_n1 * p_trans_n2 * \
                            deltakernel(G1.node[neighbor1][node_label] == G2.node[neighbor2][node_label]) * \
                            deltakernel(neighbor_n1[neighbor1][edge_label] == neighbor_n2[neighbor2][edge_label])

                        R_inf_new[node1[0]][node2[0]] += t * R_inf[neighbor1][
                            neighbor2]  # ref [1] equation (8)
        R_inf[:] = R_inf_new

    # add elements of R_inf up and calculate kernel
    for node1 in G1.nodes(data=True):
        for node2 in G2.nodes(data=True):
            s = p_init_G1 * p_init_G2 * deltakernel(
                node1[1][node_label] == node2[1][node_label])
            kernel += s * R_inf[node1[0]][node2[0]]  # ref [1] equation (6)

    return kernel
