"""
@author: linlin <jajupmochi@gmail.com>
@references:
    [1] Tamás Horváth, Thomas Gärtner, and Stefan Wrobel. Cyclic pattern kernels for predictive graph mining. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 158–167. ACM, 2004.
    [2]	Hopcroft, J.; Tarjan, R. (1973). “Efficient algorithms for graph manipulation”. Communications of the ACM 16: 372–378. doi:10.1145/362248.362272.
    [3] Finding all the elementary circuits of a directed graph. D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975. http://dx.doi.org/10.1137/0204007
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time

import networkx as nx
import numpy as np

from tqdm import tqdm


def cyclicpatternkernel(*args, node_label = 'atom', edge_label = 'bond_type', labeled = True, cycle_bound = None):
    """Calculate cyclic pattern graph kernels between graphs.
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
    depth : integer
        Depth of search. Longest length of paths.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the path kernel up to d between 2 praphs.
    """
    Gn = args[0] if len(args) == 1 else [args[0], args[1]] # arrange all graphs in a list
    Kmatrix = np.zeros((len(Gn), len(Gn)))

    start_time = time.time()

    # get all cyclic and tree patterns of all graphs before calculating kernels to save time, but this may consume a lot of memory for large dataset.
    all_patterns = [ get_patterns(Gn[i], node_label=node_label, edge_label = edge_label, labeled = labeled, cycle_bound = cycle_bound)
        for i in tqdm(range(0, len(Gn)), desc='retrieve patterns', file=sys.stdout) ]

    for i in tqdm(range(0, len(Gn)), desc='calculate kernels', file=sys.stdout):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _cyclicpatternkernel_do(all_patterns[i], all_patterns[j])
            Kmatrix[j][i] = Kmatrix[i][j]

    run_time = time.time() - start_time
    print("\n --- kernel matrix of cyclic pattern kernel of size %d built in %s seconds ---" % (len(Gn), run_time))

    return Kmatrix, run_time


def _cyclicpatternkernel_do(patterns1, patterns2):
    """Calculate path graph kernels up to depth d between 2 graphs.

    Parameters
    ----------
    paths1, paths2 : list
        List of paths in 2 graphs, where for unlabeled graphs, each path is represented by a list of nodes; while for labeled graphs, each path is represented by a string consists of labels of nodes and edges on that path.
    k_func : function
        A kernel function used using different notions of fingerprint similarity.
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
    return len(set(patterns1) & set(patterns2))


def get_patterns(G, node_label = 'atom', edge_label = 'bond_type', labeled = True, cycle_bound = None):
    """Find all cyclic and tree patterns in a graph.

    Parameters
    ----------
    G : NetworkX graphs
        The graph in which paths are searched.
    length : integer
        The maximum length of paths.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    labeled : boolean
        Whether the graphs are labeled. The default is True.

    Return
    ------
    path : list
        List of paths retrieved, where for unlabeled graphs, each path is represented by a list of nodes; while for labeled graphs, each path is represented by a string consists of labels of nodes and edges on that path.
    """
    number_simplecycles = 0
    bridges = nx.Graph()
    patterns = []

    bicomponents = nx.biconnected_component_subgraphs(G) # all biconnected components of G. this function use algorithm in reference [2], which (i guess) is slightly different from the one used in paper [1]
    for subgraph in bicomponents:
        if nx.number_of_edges(subgraph) > 1:
            simple_cycles = list(nx.simple_cycles(G.to_directed())) # all simple cycles in biconnected components. this function use algorithm in reference [3], which has time complexity O((n+e)(N+1)) for n nodes, e edges and N simple cycles. Which might be slower than the algorithm applied in paper [1]
            if cycle_bound != None and len(simple_cycles) > cycle_bound - number_simplecycles: # in paper [1], when applying another algorithm (subroutine RT), this becomes len(simple_cycles) == cycle_bound - number_simplecycles + 1, check again.
                return []
            else:

                # calculate canonical representation for each simple cycle
                all_canonkeys = []
                for cycle in simple_cycles:
                    canonlist = [ G.node[node][node_label] + G[node][cycle[cycle.index(node) + 1]][edge_label] for node in cycle[:-1] ]
                    canonkey = ''.join(canonlist)
                    canonkey = canonkey if canonkey < canonkey[::-1] else canonkey[::-1]
                    for i in range(1, len(cycle[:-1])):
                        canonlist = [ G.node[node][node_label] + G[node][cycle[cycle.index(node) + 1]][edge_label] for node in cycle[i:-1] + cycle[:i] ]
                        canonkey_t = ''.join(canonlist)
                        canonkey_t = canonkey_t if canonkey_t < canonkey_t[::-1] else canonkey_t[::-1]
                        canonkey = canonkey if canonkey < canonkey_t else canonkey_t
                    all_canonkeys.append(canonkey)

                patterns = list(set(patterns) | set(all_canonkeys))
                number_simplecycles += len(simple_cycles)
        else:
            bridges.add_edges_from(subgraph.edges(data=True))

    # calculate canonical representation for each connected component in bridge set
    components = list(nx.connected_component_subgraphs(bridges)) # all connected components in the bridge
    tree_patterns = []
    for tree in components:
        break



    # patterns += pi(bridges)
    return patterns
