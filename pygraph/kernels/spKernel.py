"""
@author: linlin
@references: Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
import pathlib
sys.path.insert(0, "../")
from tqdm import tqdm
import time

import networkx as nx
import numpy as np

from pygraph.utils.utils import getSPGraph
from pygraph.utils.graphdataset import get_dataset_attributes


def spkernel(*args, node_label='atom', edge_weight=None):
    """Calculate shortest-path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    edge_weight : string
        Edge attribute corresponding to the edge weight.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the sp kernel between 2 praphs.
    """
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    try:
        some_weight = list(
            nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
        weight = edge_label if isinstance(some_weight, float) or isinstance(
            some_weight, int) else None
    except:
        weight = None
    ds_attrs = get_dataset_attributes(
        Gn, attr_names=['node_labeled'], node_label=node_label)

    start_time = time.time()

    # get shortest path graphs of Gn
    Gn = [
        getSPGraph(G, edge_weight=edge_weight)
        for G in tqdm(Gn, desc='getting sp graphs', file=sys.stdout)
    ]

    pbar = tqdm(
        total=((len(Gn) + 1) * len(Gn) / 2),
        desc='calculating kernels',
        file=sys.stdout)
    if ds_attrs['node_labeled']:
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                for e1 in Gn[i].edges(data=True):
                    for e2 in Gn[j].edges(data=True):
                        # cost of a node to itself equals to 0, cost between two disconnected nodes is Inf.
                        if e1[2]['cost'] != 0 and e1[2] != np.Inf and e1[2]['cost'] == e2[2]['cost'] and {
                                Gn[i].nodes[e1[0]][node_label],
                                Gn[i].nodes[e1[1]][node_label]
                        } == {
                                Gn[j].nodes[e2[0]][node_label],
                                Gn[j].nodes[e2[1]][node_label]
                        }:
                            Kmatrix[i][j] += 1
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)
    else:
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                # kernel_t = [ e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])) \
                #     for e1 in Sn[i].edges(data = True) for e2 in Sn[j].edges(data = True) ]
                # Kmatrix[i][j] = np.sum(kernel_t)
                # Kmatrix[j][i] = Kmatrix[i][j]

                for e1 in Gn[i].edges(data=True):
                    for e2 in Gn[j].edges(data=True):
                        if e1[2]['cost'] != 0 and e1[2] != np.Inf and e1[2]['cost'] == e2[2]['cost']:
                            Kmatrix[i][j] += 1
                Kmatrix[j][i] = Kmatrix[i][j]
                pbar.update(1)

    run_time = time.time() - start_time
    print(
        "--- shortest path kernel matrix of size %d built in %s seconds ---" %
        (len(Gn), run_time))

    return Kmatrix, run_time
