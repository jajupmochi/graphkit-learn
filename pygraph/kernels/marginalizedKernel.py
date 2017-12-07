import sys
import pathlib
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.deltaKernel import deltakernel

def marginalizedkernel(*args):
    """Calculate marginalized graph kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    p_quit : integer
        the termination probability in the random walks generating step
    itr : integer
        time of iterations to calculate R_inf
        
    Return
    ------
    Kmatrix/Kernel : Numpy matrix/int
        Kernel matrix, each element of which is the marginalized kernel between 2 praphs. / Marginalized Kernel between 2 graphs.
        
    References
    ----------
    [1] H. Kashima, K. Tsuda, and A. Inokuchi. Marginalized kernels between labeled graphs. In Proceedings of the 20th International Conference on Machine Learning, Washington, DC, United States, 2003.
    """
    if len(args) == 3: # for a list of graphs
        Gn = args[0]

        Kmatrix = np.zeros((len(Gn), len(Gn)))

        start_time = time.time()
        
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                Kmatrix[i][j] = _marginalizedkernel_do(Gn[i], Gn[j], args[1], args[2])
                Kmatrix[j][i] = Kmatrix[i][j]
                
        print("\n --- marginalized kernel matrix of size %d built in %s seconds ---" % (len(Gn), (time.time() - start_time)))
        
        return Kmatrix
        
    else: # for only 2 graphs
        
        start_time = time.time()
        
        kernel = _marginalizedkernel_do(args[0], args[1], args[2], args[3])

        print("\n --- marginalized kernel built in %s seconds ---" % (time.time() - start_time))
        
        return kernel

    
def _marginalizedkernel_do(G1, G2, p_quit, itr):
    """Calculate marginalized graph kernels between 2 graphs.
    
    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    p_quit : integer
        the termination probability in the random walks generating step
    itr : integer
        time of iterations to calculate R_inf
        
    Return
    ------
    Kernel : int
        Marginalized Kernel between 2 graphs.
    """
    # init parameters
    kernel = 0
    num_nodes_G1 = nx.number_of_nodes(G1)
    num_nodes_G2 = nx.number_of_nodes(G2)
    p_init_G1 = 1 / num_nodes_G1 # the initial probability distribution in the random walks generating step (uniform distribution over |G|)
    p_init_G2 = 1 / num_nodes_G2

    q = p_quit * p_quit
    r1 = q

    # initial R_inf
    R_inf = np.zeros([num_nodes_G1, num_nodes_G2]) # matrix to save all the R_inf for all pairs of nodes

    # calculate R_inf with a simple interative method
    for i in range(1, itr):
        R_inf_new = np.zeros([num_nodes_G1, num_nodes_G2])
        R_inf_new.fill(r1)

        # calculate R_inf for each pair of nodes
        for node1 in G1.nodes(data = True):
            neighbor_n1 = G1[node1[0]]
            p_trans_n1 = (1 - p_quit) / len(neighbor_n1) # the transition probability distribution in the random walks generating step (uniform distribution over the vertices adjacent to the current vertex)
            for node2 in G2.nodes(data = True):
                neighbor_n2 = G2[node2[0]]
                p_trans_n2 = (1 - p_quit) / len(neighbor_n2)    

                for neighbor1 in neighbor_n1:
                    for neighbor2 in neighbor_n2:

                        t = p_trans_n1 * p_trans_n2 * \
                            deltakernel(G1.node[neighbor1]['label'] == G2.node[neighbor2]['label']) * \
                            deltakernel(neighbor_n1[neighbor1]['label'] == neighbor_n2[neighbor2]['label'])
                        R_inf_new[node1[0]][node2[0]] += t * R_inf[neighbor1][neighbor2] # ref [1] equation (8)

        R_inf[:] = R_inf_new

    # add elements of R_inf up and calculate kernel
    for node1 in G1.nodes(data = True):
        for node2 in G2.nodes(data = True):                
            s = p_init_G1 * p_init_G2 * deltakernel(node1[1]['label'] == node2[1]['label'])
            kernel += s * R_inf[node1[0]][node2[0]] # ref [1] equation (6)

    return kernel