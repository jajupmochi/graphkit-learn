import sys
import pathlib
sys.path.insert(0, "../")


import networkx as nx
import numpy as np
import time

from pygraph.utils.utils import getSPGraph


def spkernel(*args):
    """Calculate shortest-path kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
        
    Return
    ------
    Kmatrix/Kernel : Numpy matrix/int
        Kernel matrix, each element of which is the sp kernel between 2 praphs. / SP Kernel between 2 graphs.
        
    References
    ----------
    [1] Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
    """
    if len(args) == 1: # for a list of graphs
        Gn = args[0]
        
        Kmatrix = np.zeros((len(Gn), len(Gn)))
    
        Sn = [] # get shortest path graphs of Gn
        for i in range(0, len(Gn)):
            Sn.append(getSPGraph(Gn[i]))

        start_time = time.time()
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                for e1 in Sn[i].edges(data = True):
                    for e2 in Sn[j].edges(data = True):          
                        if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                            Kmatrix[i][j] += 1
                            Kmatrix[j][i] += (0 if i == j else 1)

        print("--- shortest path kernel matrix of size %d built in %s seconds ---" % (len(Gn), (time.time() - start_time)))
        
        return Kmatrix
        
    else: # for only 2 graphs
        G1 = getSPGraph(args[0])
        G2 = getSPGraph(args[1])
        
        kernel = 0
        
        start_time = time.time()
        for e1 in G1.edges(data = True):
            for e2 in G2.edges(data = True):          
                if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                    kernel += 1

        print("--- shortest path kernel built in %s seconds ---" % (time.time() - start_time))
        
        return kernel