import sys
import pathlib
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.deltaKernel import deltakernel

def pathkernel(*args):
    """Calculate mean average path kernels between graphs.
    
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
        Kernel matrix, each element of which is the path kernel between 2 praphs. / Path Kernel between 2 graphs.
        
    References
    ----------
    [1] Suard F, Rakotomamonjy A, Bensrhair A. Kernel on Bag of Paths For Measuring Similarity of Shapes. InESANN 2007 Apr 25 (pp. 355-360).
    """
    if len(args) == 1: # for a list of graphs
        Gn = args[0]
        
        Kmatrix = np.zeros((len(Gn), len(Gn)))

        start_time = time.time()
        
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _pathkernel_do(Gn[i], Gn[j])
                Kmatrix[j][i] = Kmatrix[i][j]

        print("\n --- mean average path kernel matrix of size %d built in %s seconds ---" % (len(Gn), (time.time() - start_time)))
        
        return Kmatrix
        
    else: # for only 2 graphs
        start_time = time.time()
        
        kernel = _pathkernel_do(args[0], args[1])

        print("\n --- mean average path kernel built in %s seconds ---" % (time.time() - start_time))
        
        return kernel
    
    
def _pathkernel_do(G1, G2):
    """Calculate mean average path kernels between 2 graphs.
    
    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
        
    Return
    ------
    Kernel : int
        Path Kernel between 2 graphs.
    """
    # calculate shortest paths for both graphs
    sp1 = []
    num_nodes = G1.number_of_nodes()
    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
                sp1.append(nx.shortest_path(G1, node1, node2, weight = 'cost'))
                
    sp2 = []
    num_nodes = G2.number_of_nodes()
    for node1 in range(num_nodes):
        for node2 in range(node1 + 1, num_nodes):
                sp2.append(nx.shortest_path(G2, node1, node2, weight = 'cost'))

    # calculate kernel
    kernel = 0
    for path1 in sp1:
        for path2 in sp2:
            if len(path1) == len(path2):
                kernel_path = deltakernel(G1.node[path1[0]]['label'] == G2.node[path2[0]]['label'])
                if kernel_path:
                    for i in range(1, len(path1)):
                         # kernel = 1 if all corresponding nodes and edges in the 2 paths have same labels, otherwise 0
                        kernel_path *= deltakernel(G1[path1[i - 1]][path1[i]]['label'] == G2[path2[i - 1]][path2[i]]['label']) * deltakernel(G1.node[path1[i]]['label'] == G2.node[path2[i]]['label'])
                    kernel += kernel_path # add up kernels of all paths

    kernel = kernel / (len(sp1) * len(sp2)) # calculate mean average
    
    return kernel