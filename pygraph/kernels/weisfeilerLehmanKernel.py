"""
@author: linlin
@references:
    [1] Shervashidze N, Schweitzer P, Leeuwen EJ, Mehlhorn K, Borgwardt KM. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research. 2011;12(Sep):2539-61.
"""

import sys
import pathlib
from collections import Counter
sys.path.insert(0, "../")

import networkx as nx
import numpy as np
import time

from pygraph.kernels.pathKernel import pathkernel

def weisfeilerlehmankernel(*args, node_label = 'atom', edge_label = 'bond_type', height = 0, base_kernel = 'subtree'):
    """Calculate Weisfeiler-Lehman kernels between graphs.
    
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
    height : int
        subtree height    
    base_kernel : string
        base kernel used in each iteration of WL kernel. The default base kernel is subtree kernel. For user-defined kernel, base_kernel is the name of the base kernel function used in each iteration of WL kernel. This function returns a Numpy matrix, each element of which is the user-defined Weisfeiler-Lehman kernel between 2 praphs.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.

    Notes
    -----
    This function now supports WL subtree kernel, WL shortest path kernel and WL edge kernel.
    """
    base_kernel = base_kernel.lower()
    Gn = args[0] if len(args) == 1 else [args[0], args[1]] # arrange all graphs in a list
    Kmatrix = np.zeros((len(Gn), len(Gn)))

    start_time = time.time()

    # for WL subtree kernel
    if base_kernel == 'subtree':           
        Kmatrix = _wl_subtreekernel_do(args[0], node_label, edge_label, height)

    # for WL shortest path kernel
    elif base_kernel == 'sp':
        Kmatrix = _wl_spkernel_do(args[0], node_label, edge_label, height)

    # for WL edge kernel
    elif base_kernel == 'edge':
        Kmatrix = _wl_edgekernel_do(args[0], node_label, edge_label, height)

    # for user defined base kernel
    else:
        Kmatrix = _wl_userkernel_do(args[0], node_label, edge_label, height, base_kernel)

    run_time = time.time() - start_time
    print("\n --- Weisfeiler-Lehman %s kernel matrix of size %d built in %s seconds ---" % (base_kernel, len(args[0]), run_time))

    return Kmatrix, run_time



def _wl_subtreekernel_do(Gn, node_label, edge_label, height):
    """Calculate Weisfeiler-Lehman subtree kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.       
    node_label : string
        node attribute used as label.
    edge_label : string
        edge attribute used as label.      
    height : int
        subtree height.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
    """
    height = int(height)
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    all_num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs

    # initial for height = 0
    all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
    all_num_of_each_label = [] # number of occurence of each label in each graph in this iteration
    all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
    num_of_labels_occured = all_num_of_labels_occured # number of the set of letters that occur before as node labels at least once in all graphs

    # for each graph
    for G in Gn:
        # get the set of original labels
        labels_ori = list(nx.get_node_attributes(G, node_label).values())
        all_labels_ori.update(labels_ori)
        num_of_each_label = dict(Counter(labels_ori)) # number of occurence of each label in graph
        all_num_of_each_label.append(num_of_each_label)
        num_of_labels = len(num_of_each_label) # number of all unique labels

        all_labels_ori.update(labels_ori)

    all_num_of_labels_occured += len(all_labels_ori)

    # calculate subtree kernel with the 0th iteration and add it to the final kernel
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            labels = set(list(all_num_of_each_label[i].keys()) + list(all_num_of_each_label[j].keys()))
            vector1 = np.matrix([ (all_num_of_each_label[i][label] if (label in all_num_of_each_label[i].keys()) else 0) for label in labels ])
            vector2 = np.matrix([ (all_num_of_each_label[j][label] if (label in all_num_of_each_label[j].keys()) else 0) for label in labels ])
            Kmatrix[i][j] += np.dot(vector1, vector2.transpose())
            Kmatrix[j][i] = Kmatrix[i][j]

    # iterate each height
    for h in range(1, height + 1):
        all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
        num_of_labels_occured = all_num_of_labels_occured # number of the set of letters that occur before as node labels at least once in all graphs
        all_labels_ori = set()
        all_num_of_each_label = []

        # for each graph
        for idx, G in enumerate(Gn):

            set_multisets = []
            for node in G.nodes(data = True):
                # Multiset-label determination.
                multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
                # sorting each multiset
                multiset.sort()
                multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
                set_multisets.append(multiset)

            # label compression
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

            all_set_compressed.update(set_compressed)

            # relabel nodes
            for node in G.nodes(data = True):
                node[1][node_label] = set_compressed[set_multisets[node[0]]]

            # get the set of compressed labels
            labels_comp = list(nx.get_node_attributes(G, node_label).values())
            all_labels_ori.update(labels_comp)
            num_of_each_label = dict(Counter(labels_comp))
            all_num_of_each_label.append(num_of_each_label)

        all_num_of_labels_occured += len(all_labels_ori)

        # calculate subtree kernel with h iterations and add it to the final kernel
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                labels = set(list(all_num_of_each_label[i].keys()) + list(all_num_of_each_label[j].keys()))
                vector1 = np.matrix([ (all_num_of_each_label[i][label] if (label in all_num_of_each_label[i].keys()) else 0) for label in labels ])
                vector2 = np.matrix([ (all_num_of_each_label[j][label] if (label in all_num_of_each_label[j].keys()) else 0) for label in labels ])
                Kmatrix[i][j] += np.dot(vector1, vector2.transpose())
                Kmatrix[j][i] = Kmatrix[i][j]

    return Kmatrix


def _wl_spkernel_do(Gn, node_label, edge_label, height):
    """Calculate Weisfeiler-Lehman shortest path kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.       
    node_label : string
        node attribute used as label.      
    edge_label : string
        edge attribute used as label.       
    height : int
        subtree height.
        
    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
    """
    from pygraph.utils.utils import getSPGraph
      
    # init.
    height = int(height)
    Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel

    Gn = [ getSPGraph(G, edge_weight = edge_label) for G in Gn ] # get shortest path graphs of Gn
    
    # initial for height = 0
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            for e1 in Gn[i].edges(data = True):
                for e2 in Gn[j].edges(data = True):          
                    if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                        Kmatrix[i][j] += 1
            Kmatrix[j][i] = Kmatrix[i][j]
            
    # iterate each height
    for h in range(1, height + 1):
        all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
        num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
        for G in Gn: # for each graph
            set_multisets = []
            for node in G.nodes(data = True):
                # Multiset-label determination.
                multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
                # sorting each multiset
                multiset.sort()
                multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
                set_multisets.append(multiset)          

            # label compression
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

            all_set_compressed.update(set_compressed)
            
            # relabel nodes
            for node in G.nodes(data = True):
                node[1][node_label] = set_compressed[set_multisets[node[0]]]
                
        # calculate subtree kernel with h iterations and add it to the final kernel
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                for e1 in Gn[i].edges(data = True):
                    for e2 in Gn[j].edges(data = True):          
                        if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                            Kmatrix[i][j] += 1
                Kmatrix[j][i] = Kmatrix[i][j]
        
    return Kmatrix



def _wl_edgekernel_do(Gn, node_label, edge_label, height):
    """Calculate Weisfeiler-Lehman edge kernels between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.       
    node_label : string
        node attribute used as label.      
    edge_label : string
        edge attribute used as label.       
    height : int
        subtree height.
        
    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
    """      
    # init.
    height = int(height)
    Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel
  
    # initial for height = 0
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            for e1 in Gn[i].edges(data = True):
                for e2 in Gn[j].edges(data = True):          
                    if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                        Kmatrix[i][j] += 1
            Kmatrix[j][i] = Kmatrix[i][j]
            
    # iterate each height
    for h in range(1, height + 1):
        all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
        num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
        for G in Gn: # for each graph
            set_multisets = []            
            for node in G.nodes(data = True):
                # Multiset-label determination.
                multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
                # sorting each multiset
                multiset.sort()
                multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
                set_multisets.append(multiset)          

            # label compression
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

            all_set_compressed.update(set_compressed)
            
            # relabel nodes
            for node in G.nodes(data = True):
                node[1][node_label] = set_compressed[set_multisets[node[0]]]
                
        # calculate subtree kernel with h iterations and add it to the final kernel
        for i in range(0, len(Gn)):
            for j in range(i, len(Gn)):
                for e1 in Gn[i].edges(data = True):
                    for e2 in Gn[j].edges(data = True):          
                        if e1[2][edge_label] == e2[2][edge_label] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                            Kmatrix[i][j] += 1
                Kmatrix[j][i] = Kmatrix[i][j]
        
    return Kmatrix


def _wl_userkernel_do(Gn, node_label, edge_label, height, base_kernel):
    """Calculate Weisfeiler-Lehman kernels based on user-defined kernel between graphs.
    
    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.       
    node_label : string
        node attribute used as label.      
    edge_label : string
        edge attribute used as label.       
    height : int
        subtree height.
    base_kernel : string
        Name of the base kernel function used in each iteration of WL kernel. This function returns a Numpy matrix, each element of which is the user-defined Weisfeiler-Lehman kernel between 2 praphs.
        
    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the Weisfeiler-Lehman kernel between 2 praphs.
    """      
    # init.
    height = int(height)
    Kmatrix = np.zeros((len(Gn), len(Gn))) # init kernel
  
    # initial for height = 0
    Kmatrix = base_kernel(Gn, node_label, edge_label)
            
    # iterate each height
    for h in range(1, height + 1):
        all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
        num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
        for G in Gn: # for each graph
            set_multisets = []           
            for node in G.nodes(data = True):
                # Multiset-label determination.
                multiset = [ G.node[neighbors][node_label] for neighbors in G[node[0]] ]
                # sorting each multiset
                multiset.sort()
                multiset = node[1][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
                set_multisets.append(multiset)          

            # label compression
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

            all_set_compressed.update(set_compressed)
            
            # relabel nodes
            for node in G.nodes(data = True):
                node[1][node_label] = set_compressed[set_multisets[node[0]]]
                
        # calculate kernel with h iterations and add it to the final kernel
        Kmatrix += base_kernel(Gn, node_label, edge_label)
        
    return Kmatrix
