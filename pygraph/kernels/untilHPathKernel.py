"""
@author: linlin
@references: Liva Ralaivola, Sanjay J Swamidass, Hiroto Saigo, and Pierre Baldi. Graph kernels for chemical informatics. Neural networks, 18(8):1093–1110, 2005.
"""

import sys
import pathlib
sys.path.insert(0, "../")
import time
from collections import Counter
from itertools import chain
from tqdm import tqdm

import networkx as nx
import numpy as np

from pygraph.utils.graphdataset import get_dataset_attributes


def untilhpathkernel(*args,
                     node_label='atom',
                     edge_label='bond_type',
                     depth=10,
                     k_func='tanimoto'):
    """Calculate path graph kernels up to depth d between graphs.
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
    depth : integer
        Depth of search. Longest length of paths.
    k_func : function
        A kernel function used using different notions of fingerprint similarity.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the path kernel up to d between 2 praphs.
    """
    depth = int(depth)
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    Kmatrix = np.zeros((len(Gn), len(Gn)))
    ds_attrs = get_dataset_attributes(
        Gn,
        attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
        node_label=node_label,
        edge_label=edge_label)

    start_time = time.time()

    # get all paths of all graphs before calculating kernels to save time, but this may cost a lot of memory for large datasets.
    all_paths = [
        find_all_paths_until_length(
            Gn[i],
            depth,
            ds_attrs,
            node_label=node_label,
            edge_label=edge_label) for i in tqdm(
                range(0, len(Gn)), desc='getting paths', file=sys.stdout)
    ]

    pbar = tqdm(
        total=((len(Gn) + 1) * len(Gn) / 2),
        desc='calculating kernels',
        file=sys.stdout)
    for i in range(0, len(Gn)):
        for j in range(i, len(Gn)):
            Kmatrix[i][j] = _untilhpathkernel_do(all_paths[i], all_paths[j],
                                                 k_func)
            Kmatrix[j][i] = Kmatrix[i][j]
            pbar.update(1)

    run_time = time.time() - start_time
    print(
        "\n --- kernel matrix of path kernel up to %d of size %d built in %s seconds ---"
        % (depth, len(Gn), run_time))

    return Kmatrix, run_time


def _untilhpathkernel_do(paths1, paths2, k_func):
    """Calculate path graph kernels up to depth d between 2 graphs.

    Parameters
    ----------
    paths1, paths2 : list
        List of paths in 2 graphs, where for unlabeled graphs, each path is represented by a list of nodes; while for labeled graphs, each path is represented by a string consists of labels of nodes and edges on that path.
    k_func : function
        A kernel function used using different notions of fingerprint similarity.

    Return
    ------
    kernel : float
        Treelet Kernel between 2 graphs.
    """
    all_paths = list(set(paths1 + paths2))

    if k_func == 'tanimoto':
        vector1 = [(1 if path in paths1 else 0) for path in all_paths]
        vector2 = [(1 if path in paths2 else 0) for path in all_paths]
        kernel_uv = np.dot(vector1, vector2)
        kernel = kernel_uv / (len(set(paths1)) + len(set(paths2)) - kernel_uv)

    else:  # MinMax kernel
        path_count1 = Counter(paths1)
        path_count2 = Counter(paths2)
        vector1 = [(path_count1[key] if (key in path_count1.keys()) else 0)
                   for key in all_paths]
        vector2 = [(path_count2[key] if (key in path_count2.keys()) else 0)
                   for key in all_paths]
        kernel = np.sum(np.minimum(vector1, vector2)) / \
            np.sum(np.maximum(vector1, vector2))

    return kernel


# this method find paths repetively, it could be faster.


def find_all_paths_until_length(G,
                                length,
                                ds_attrs,
                                node_label='atom',
                                edge_label='bond_type'):
    """Find all paths no longer than a certain maximum length in a graph. A recursive depth first search is applied.

    Parameters
    ----------
    G : NetworkX graphs
        The graph in which paths are searched.
    length : integer
        The maximum length of paths.
    ds_attrs: dict
        Dataset attributes.
    node_label : string
        Node attribute used as label. The default node label is atom.
    edge_label : string
        Edge attribute used as label. The default edge label is bond_type.

    Return
    ------
    path : list
        List of paths retrieved, where for unlabeled graphs, each path is represented by a list of nodes; while for labeled graphs, each path is represented by a list of strings consists of labels of nodes and/or edges on that path.
    """
    # path_l = [tuple([n]) for n in G.nodes]  # paths of length l
    # all_paths = path_l[:]
    # for l in range(1, length + 1):
    #     path_l_new = []
    #     for path in path_l:
    #         for neighbor in G[path[-1]]:
    #             if len(path) < 2 or neighbor != path[-2]:
    #                 tmp = path + (neighbor, )
    #                 if tuple(tmp[::-1]) not in path_l_new:
    #                     path_l_new.append(tuple(tmp))

    #     all_paths += path_l_new
    #     path_l = path_l_new[:]

    path_l = [[n] for n in G.nodes]  # paths of length l
    all_paths = path_l[:]
    for l in range(1, length + 1):
        path_l_new = []
        for path in path_l:
            for neighbor in G[path[-1]]:
                if len(path) < 2 or neighbor != path[-2]:
                    tmp = path + [neighbor]
                    if tmp[::-1] not in path_l_new:
                        path_l_new.append(tmp)

        all_paths += path_l_new
        path_l = path_l_new[:]

    # for i in range(0, length + 1):
    #     new_paths = find_all_paths(G, i)
    #     if new_paths == []:
    #         break
    #     all_paths.extend(new_paths)

    # consider labels
    if ds_attrs['node_labeled']:
        if ds_attrs['edge_labeled']:
            path_strs = [
                tuple(
                    list(
                        chain.from_iterable(
                            (G.node[node][node_label],
                             G[node][path[idx + 1]][edge_label])
                            for idx, node in enumerate(path[:-1]))) +
                    [G.node[path[-1]][node_label]]) for path in all_paths
            ]

            # path_strs = []
            # for path in all_paths:
            #     strlist = list(
            #         chain.from_iterable((G.node[node][node_label],
            #                              G[node][path[idx + 1]][edge_label])
            #                             for idx, node in enumerate(path[:-1])))
            #     strlist.append(G.node[path[-1]][node_label])
            #     path_strs.append(tuple(strlist))
        else:
            path_strs = [
                tuple([G.node[node][node_label] for node in path])
                for path in all_paths
            ]
        return path_strs
    else:
        if ds_attrs['edge_labeled']:
            return [
                tuple([] if len(path) == 1 else [
                    G[node][path[idx + 1]][edge_label]
                    for idx, node in enumerate(path[:-1])
                ]) for path in all_paths
            ]
        else:
            return [tuple([len(path)]) for path in all_paths]


# def find_paths(G, source_node, length):
#     """Find all paths no longer than a certain length those start from a source node. A recursive depth first search is applied.

#     Parameters
#     ----------
#     G : NetworkX graphs
#         The graph in which paths are searched.
#     source_node : integer
#         The number of the node from where all paths start.
#     length : integer
#         The length of paths.

#     Return
#     ------
#     path : list of list
#         List of paths retrieved, where each path is represented by a list of nodes.
#     """
#     return [[source_node]] if length == 0 else \
#         [[source_node] + path for neighbor in G[source_node]
#          for path in find_paths(G, neighbor, length - 1) if source_node not in path]

# def find_all_paths(G, length):
#     """Find all paths with a certain length in a graph. A recursive depth first search is applied.

#     Parameters
#     ----------
#     G : NetworkX graphs
#         The graph in which paths are searched.
#     length : integer
#         The length of paths.

#     Return
#     ------
#     path : list of list
#         List of paths retrieved, where each path is represented by a list of nodes.
#     """
#     all_paths = []
#     for node in G:
#         all_paths.extend(find_paths(G, node, length))

#     # The following process is not carried out according to the original article
#     # all_paths_r = [ path[::-1] for path in all_paths ]

#     # # For each path, two presentation are retrieved from its two extremities. Remove one of them.
#     # for idx, path in enumerate(all_paths[:-1]):
#     #     for path2 in all_paths_r[idx+1::]:
#     #         if path == path2:
#     #             all_paths[idx] = []
#     #             break

#     # return list(filter(lambda a: a != [], all_paths))
#     return all_paths
