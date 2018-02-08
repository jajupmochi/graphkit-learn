"""
@author: linlin
@references: Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
import pathlib
sys.path.insert(0, "../")


import networkx as nx
import numpy as np
import time

from pygraph.utils.utils import getSPGraph


def spkernel(*args, edge_weight = 'bond_type'):
    """Calculate shortest-path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    edge_weight : string
        edge attribute corresponding to the edge weight. The default edge weight is bond_type.

    Return
    ------
    Kmatrix/kernel : Numpy matrix/float
        Kernel matrix, each element of which is the sp kernel between 2 praphs. / SP kernel between 2 graphs.
    """
    Gn = args[0] if len(args) == 1 else [args[0], args[1]] # arrange all graphs in a list
    Kmatrix = np.zeros((len(Gn), len(Gn)))

    start_time = time.time()

    Gn = [ getSPGraph(G, edge_weight = edge_weight) for G in args[0] ] # get shortest path graphs of Gn

    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
                # kernel_t = [ e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])) \
                #     for e1 in Sn[i].edges(data = True) for e2 in Sn[j].edges(data = True) ]
                # Kmatrix[i][j] = np.sum(kernel_t)
                # Kmatrix[j][i] = Kmatrix[i][j]

            for e1 in Gn[i].edges(data = True):
                for e2 in Gn[j].edges(data = True):
                    if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                        Kmatrix[i][j] += 1
            Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print("--- shortest path kernel matrix of size %d built in %s seconds ---" % (len(Gn), run_time))

    return Kmatrix, run_time