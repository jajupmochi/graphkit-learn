import sys
import pathlib
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.spkernel import spkernel
from pygraph.kernels.pathKernel import pathkernel

# test of WL subtree kernel on many graphs

import sys
import pathlib
from collections import Counter
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.spkernel import spkernel
from pygraph.kernels.pathKernel import pathkernel

def weisfeilerlehmankernel(*args, height = 0, base_kernel = 'subtree'):
    """Calculate Weisfeiler-Lehman kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
        
    height : subtree height
    
    base_kernel : base kernel used in each iteration of WL kernel
        the default base kernel is subtree kernel
        
    Return
    ------
    Kmatrix/Kernel : Numpy matrix/int
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs. / Weisfeiler-Lehman Kernel between 2 graphs.
        
    Notes
    -----
    This function now supports WL subtree kernel and WL shortest path kernel.
    
    References
    ----------
    [1] Shervashidze N, Schweitzer P, Leeuwen EJ, Mehlhorn K, Borgwardt KM. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research. 2011;12(Sep):2539-61.
    """
    if len(args) == 1: # for a list of graphs
        start_time = time.time()
        
        # for WL subtree kernel
        if base_kernel == 'subtree':           
            Kmatrix = _wl_subtreekernel_do(args[0], height = height, base_kernel = 'subtree')
            
        # for WL edge kernel
        elif base_kernel == 'edge':
            print('edge')
            
        # for WL shortest path kernel
        elif base_kernel == 'sp':
            Gn = args[0]
            Kmatrix = np.zeros((len(Gn), len(Gn)))
            
            for i in range(0, len(Gn)):
                for j in range(i, len(Gn)):
                    Kmatrix[i][j] = _weisfeilerlehmankernel_do(Gn[i], Gn[j])
                    Kmatrix[j][i] = Kmatrix[i][j]

        print("\n --- Weisfeiler-Lehman %s kernel matrix of size %d built in %s seconds ---" % (base_kernel, len(args[0]), (time.time() - start_time)))
        
        return Kmatrix
        
    else: # for only 2 graphs
        
        start_time = time.time()
        
        # for WL subtree kernel
        if base_kernel == 'subtree':
            
            args = [args[0], args[1]]
            kernel = _wl_subtreekernel_do(args, height = height, base_kernel = 'subtree')
            
        # for WL edge kernel
        elif base_kernel == 'edge':
            print('edge')
            
        # for WL shortest path kernel
        elif base_kernel == 'sp':
            

            kernel = _pathkernel_do(args[0], args[1])

        print("\n --- Weisfeiler-Lehman %s kernel built in %s seconds ---" % (base_kernel, time.time() - start_time))
        
        return kernel
    
    
def _wl_subtreekernel_do(*args, height = 0, base_kernel = 'subtree'):
    """Calculate Weisfeiler-Lehman subtree kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
        
    Return
    ------
    Kmatrix/Kernel : Numpy matrix/int
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
    """
    
    Gn = args[0]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    all_num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
    
    # iterate each height
    for h in range(height + 1):
        all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
        all_num_of_each_label = [] # number of occurence of each label in each graph in this iteration
        all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
        num_of_labels_occured = all_num_of_labels_occured # number of the set of letters that occur before as node labels at least once in all graphs
        
        # for each graph
        for idx, G in enumerate(Gn):
            # get the set of original labels
            labels_ori = list(nx.get_node_attributes(G, 'label').values())
            num_of_each_label = dict(Counter(labels_ori)) # number of occurence of each label in graph
            num_of_labels = len(num_of_each_label) # number of all unique labels
            
            all_labels_ori.update(labels_ori)
            # num_of_labels_occured += num_of_labels #@todo not precise
            num_of_labels_occured = all_num_of_labels_occured + len(all_labels_ori) + len(all_set_compressed)
            
            set_multisets = []
            for node in G.nodes(data = True):
                # Multiset-label determination.
                multiset = [ G.node[neighbors]['label'] for neighbors in G[node[0]] ]
                # sorting each multiset
                multiset.sort()
                multiset = node[1]['label'] + ''.join(multiset) # concatenate to a string and add the prefix 
                set_multisets.append(multiset)

            # label compression
        #     set_multisets.sort() # this is unnecessary
            set_unique = list(set(set_multisets)) # set of unique multiset labels
            # a dictionary mapping original labels to new ones. 
            set_compressed = {}
            # if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label 
            for value in set_unique:
                if value in all_set_compressed.keys():
                    set_compressed.update({ value : all_set_compressed[value] })
                else:
                    set_compressed.update({ value : str(num_of_labels_occured + 1) })
                    num_of_labels_occured += 1
#             set_compressed = { value : (all_set_compressed[value] if value in all_set_compressed.keys() else str(set_unique.index(value) + num_of_labels_occured + 1)) for value in set_unique }
            
            all_set_compressed.update(set_compressed)
#             num_of_labels_occured += len(set_compressed) #@todo not precise

            # relabel nodes
        #     nx.relabel_nodes(G, set_compressed, copy = False)
            for node in G.nodes(data = True):
                node[1]['label'] = set_compressed[set_multisets[node[0]]]

            # get the set of compressed labels
            labels_comp = list(nx.get_node_attributes(G, 'label').values())
            num_of_each_label.update(dict(Counter(labels_comp)))
            all_num_of_each_label.append(num_of_each_label)
        
        # calculate subtree kernel with h iterations and add it to the final kernel
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                labels = set(list(all_num_of_each_label[i].keys()) + list(all_num_of_each_label[j].keys()))
                vector1 = np.matrix([ (all_num_of_each_label[i][label] if (label in all_num_of_each_label[i].keys()) else 0) for label in labels ])
                vector2 = np.matrix([ (all_num_of_each_label[j][label] if (label in all_num_of_each_label[j].keys()) else 0) for label in labels ])
                Kmatrix[i][j] += np.dot(vector1, vector2.transpose())
                Kmatrix[j][i] = Kmatrix[i][j]
                    
        all_num_of_labels_occured += len(all_labels_ori)

    return Kmatrix
    
    
def _weisfeilerlehmankernel_do(G1, G2):
    """Calculate Weisfeiler-Lehman kernels between 2 graphs. This kernel use shortest path kernel to calculate kernel between two graphs in each iteration.
    
    Parameters
    ----------
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
        
    Return
    ------
    Kernel : int
        Weisfeiler-Lehman Kernel between 2 graphs.
    """
    
    # init.
    kernel = 0 # init kernel
    num_nodes1 = G1.number_of_nodes()
    num_nodes2 = G2.number_of_nodes()
    height = 12 #min(num_nodes1, num_nodes2)) #Q how to determine the upper bound of the height?
    
    # the first iteration.
#     labelset1 = { G1.nodes(data = True)[i]['label'] for i in range(num_nodes1) }
#     labelset2 = { G2.nodes(data = True)[i]['label'] for i in range(num_nodes2) }
    kernel += pathkernel(G1, G2) # change your base kernel here (and one more below)
    
    for h in range(0, height):
#         if labelset1 != labelset2:
#             break

        # Weisfeiler-Lehman test of graph isomorphism.
        relabel(G1)
        relabel(G2)

        # calculate kernel
        kernel += pathkernel(G1, G2) # change your base kernel here (and one more before)

        # get label sets of both graphs
#         labelset1 = { G1.nodes(data = True)[i]['label'] for i in range(num_nodes1) }
#         labelset2 = { G2.nodes(data = True)[i]['label'] for i in range(num_nodes2) }
    
    return kernel


def relabel(G):
    '''
    Relabel nodes in graph G in one iteration of the 1-dim. WL test of graph isomorphism.
    
    Parameters
    ----------
    G : NetworkX graph
        The graphs whose nodes are relabeled.
    '''
    
    # get the set of original labels
    labels_ori = list(nx.get_node_attributes(G, 'label').values())
    num_of_each_label = dict(Counter(labels_ori))
    num_of_labels = len(num_of_each_label)
    
    set_multisets = []
    for node in G.nodes(data = True):
        # Multiset-label determination.
        multiset = [ G.node[neighbors]['label'] for neighbors in G[node[0]] ]
        # sorting each multiset
        multiset.sort()
        multiset = node[1]['label'] + ''.join(multiset) # concatenate to a string and add the prefix 
        set_multisets.append(multiset)
        
    # label compression
#     set_multisets.sort() # this is unnecessary
    set_unique = list(set(set_multisets)) # set of unique multiset labels
    set_compressed = { value : str(set_unique.index(value) + num_of_labels + 1) for value in set_unique } # assign new labels
    
    # relabel nodes
#     nx.relabel_nodes(G, set_compressed, copy = False)
    for node in G.nodes(data = True):
        node[1]['label'] = set_compressed[set_multisets[node[0]]]

    # get the set of compressed labels
    labels_comp = list(nx.get_node_attributes(G, 'label').values())
    num_of_each_label.update(dict(Counter(labels_comp)))