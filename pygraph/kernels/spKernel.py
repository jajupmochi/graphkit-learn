"""
@author: linlin
@references: Borgwardt KM, Kriegel HP. Shortest-path kernels on graphs. InData Mining, Fifth IEEE International Conference on 2005 Nov 27 (pp. 8-pp). IEEE.
"""

import sys
import pathlib
sys.path.insert(0, "../")
from tqdm import tqdm
import time
from itertools import combinations, combinations_with_replacement, product
from functools import partial
from joblib import Parallel, delayed
from multiprocessing import Pool

import networkx as nx
import numpy as np

from pygraph.utils.utils import getSPGraph
from pygraph.utils.graphdataset import get_dataset_attributes


def spkernel(*args,
             node_label='atom',
             edge_weight=None,
             node_kernels=None,
             n_jobs=None):
    """Calculate shortest-path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    edge_weight : string
        Edge attribute name corresponding to the edge weight.
    node_kernels: dict
        A dictionary of kernel functions for nodes, including 3 items: 'symb' for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix' for both labels. The first 2 functions take two node labels as parameters, and the 'mix' function takes 4 parameters, a symbolic and a non-symbolic label for each the two nodes. Each label is in form of 2-D dimension array (n_samples, n_features). Each function returns an number as the kernel value. Ignored when nodes are unlabeled.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the sp kernel between 2 praphs.
    """
    # pre-process
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]

    weight = None
    if edge_weight == None:
        print('\n None edge weight specified. Set all weight to 1.\n')
    else:
        try:
            some_weight = list(
                nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
            if isinstance(some_weight, float) or isinstance(some_weight, int):
                weight = edge_weight
            else:
                print(
                    '\n Edge weight with name %s is not float or integer. Set all weight to 1.\n'
                    % edge_weight)
        except:
            print(
                '\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n'
                % edge_weight)
    ds_attrs = get_dataset_attributes(
        Gn,
        attr_names=['node_labeled', 'node_attr_dim', 'is_directed'],
        node_label=node_label)

    # remove graphs with no edges, as no sp can be found in their structures, so the kernel between such a graph and itself will be zero.
    len_gn = len(Gn)
    Gn = [(idx, G) for idx, G in enumerate(Gn) if nx.number_of_edges(G) != 0]
    idx = [G[0] for G in Gn]
    Gn = [G[1] for G in Gn]
    if len(Gn) != len_gn:
        print('\n %d graphs are removed as they don\'t contain edges.\n' %
              (len_gn - len(Gn)))

    start_time = time.time()

    pool = Pool(n_jobs)
    # get shortest path graphs of Gn
    getsp_partial = partial(wrap_getSPGraph, Gn, edge_weight)
    if len(Gn) < 100:
        # use default chunksize as pool.map when iterable is less than 100
        chunksize, extra = divmod(len(Gn), n_jobs * 4)
        if extra:
            chunksize += 1
    else:
        chunksize = 100
    # chunksize = 300  # int(len(list(itr)) / n_jobs)
    for i, g in tqdm(
            pool.imap_unordered(getsp_partial, range(0, len(Gn)), chunksize),
            desc='getting sp graphs',
            file=sys.stdout):
        Gn[i] = g

    # # ---- use pool.map to parallel ----
    # result_sp = pool.map(getsp_partial, range(0, len(Gn)))
    # for i in result_sp:
    #     Gn[i[0]] = i[1]
    # or
    # getsp_partial = partial(wrap_getSPGraph, Gn, edge_weight)
    # for i, g in tqdm(
    #         pool.map(getsp_partial, range(0, len(Gn))),
    #         desc='getting sp graphs',
    #         file=sys.stdout):
    #     Gn[i] = g

    # # ---- only for the Fast Computation of Shortest Path Kernel (FCSP)
    # sp_ml = [0] * len(Gn)  # shortest path matrices
    # for i in result_sp:
    #     sp_ml[i[0]] = i[1]
    # edge_x_g = [[] for i in range(len(sp_ml))]
    # edge_y_g = [[] for i in range(len(sp_ml))]
    # edge_w_g = [[] for i in range(len(sp_ml))]
    # for idx, item in enumerate(sp_ml):
    #     for i1 in range(len(item)):
    #         for i2 in range(i1 + 1, len(item)):
    #             if item[i1, i2] != np.inf:
    #                 edge_x_g[idx].append(i1)
    #                 edge_y_g[idx].append(i2)
    #                 edge_w_g[idx].append(item[i1, i2])
    # print(len(edge_x_g[0]))
    # print(len(edge_y_g[0]))
    # print(len(edge_w_g[0]))

    Kmatrix = np.zeros((len(Gn), len(Gn)))

    # ---- use pool.imap_unordered to parallel and track progress. ----
    do_partial = partial(spkernel_do, Gn, ds_attrs, node_label, node_kernels)
    itr = combinations_with_replacement(range(0, len(Gn)), 2)
    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
    if len_itr < 100:
        chunksize, extra = divmod(len_itr, n_jobs * 4)
        if extra:
            chunksize += 1
    else:
        chunksize = 100
    for i, j, kernel in tqdm(
            pool.imap_unordered(do_partial, itr, chunksize),
            desc='calculating kernels',
            file=sys.stdout):
        Kmatrix[i][j] = kernel
        Kmatrix[j][i] = kernel
    pool.close()
    pool.join()

    # # ---- use pool.map to parallel. ----
    # # result_perf = pool.map(do_partial, itr)
    # do_partial = partial(spkernel_do, Gn, ds_attrs, node_label, node_kernels)
    # itr = combinations_with_replacement(range(0, len(Gn)), 2)
    # for i, j, kernel in tqdm(
    #         pool.map(do_partial, itr), desc='calculating kernels',
    #         file=sys.stdout):
    #     Kmatrix[i][j] = kernel
    #     Kmatrix[j][i] = kernel
    # pool.close()
    # pool.join()

    # # ---- use joblib.Parallel to parallel and track progress. ----
    # result_perf = Parallel(
    #     n_jobs=n_jobs, verbose=10)(
    #         delayed(do_partial)(ij)
    #         for ij in combinations_with_replacement(range(0, len(Gn)), 2))
    # result_perf = [
    #     do_partial(ij)
    #     for ij in combinations_with_replacement(range(0, len(Gn)), 2)
    # ]
    # for i in result_perf:
    #     Kmatrix[i[0]][i[1]] = i[2]
    #     Kmatrix[i[1]][i[0]] = i[2]

    # # ---- direct running, normally use single CPU core. ----
    # itr = combinations_with_replacement(range(0, len(Gn)), 2)
    # for gs in tqdm(itr, desc='calculating kernels', file=sys.stdout):
    #     i, j, kernel = spkernel_do(Gn, ds_attrs, node_label, node_kernels, gs)
    #     Kmatrix[i][j] = kernel
    #     Kmatrix[j][i] = kernel

    run_time = time.time() - start_time
    print(
        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time, idx


def spkernel_do(Gn, ds_attrs, node_label, node_kernels, ij):

    i = ij[0]
    j = ij[1]
    g1 = Gn[i]
    g2 = Gn[j]
    Kmatrix = 0

    try:
        # compute shortest path matrices first, method borrowed from FCSP.
        if ds_attrs['node_labeled']:
            # node symb and non-synb labeled
            if ds_attrs['node_attr_dim'] > 0:
                kn = node_kernels['mix']
                vk_dict = {}  # shortest path matrices dict
                for n1, n2 in product(
                        g1.nodes(data=True), g2.nodes(data=True)):
                    vk_dict[(n1[0], n2[0])] = kn(
                        n1[1][node_label], n2[1][node_label],
                        [n1[1]['attributes']], [n2[1]['attributes']])
            # node symb labeled
            else:
                kn = node_kernels['symb']
                vk_dict = {}  # shortest path matrices dict
                for n1 in g1.nodes(data=True):
                    for n2 in g2.nodes(data=True):
                        vk_dict[(n1[0], n2[0])] = kn(n1[1][node_label],
                                                     n2[1][node_label])
        else:
            # node non-synb labeled
            if ds_attrs['node_attr_dim'] > 0:
                kn = node_kernels['nsymb']
                vk_dict = {}  # shortest path matrices dict
                for n1 in g1.nodes(data=True):
                    for n2 in g2.nodes(data=True):
                        vk_dict[(n1[0], n2[0])] = kn([n1[1]['attributes']],
                                                     [n2[1]['attributes']])
            # node unlabeled
            else:
                for e1, e2 in product(
                        Gn[i].edges(data=True), Gn[j].edges(data=True)):
                    if e1[2]['cost'] == e2[2]['cost']:
                        Kmatrix += 1
                return i, j, Kmatrix

        # compute graph kernels
        if ds_attrs['is_directed']:
            for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
                if e1[2]['cost'] == e2[2]['cost']:
                    # each edge walk is counted twice, starting from both its extreme nodes.
                    nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(e1[1],
                                                                   e2[1])]
                    kn1 = nk11 * nk22
                    Kmatrix += kn1 + kn2
        else:
            for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
                if e1[2]['cost'] == e2[2]['cost']:
                    # each edge walk is counted twice, starting from both its extreme nodes.
                    nk11, nk12, nk21, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(
                        e1[0], e2[1])], vk_dict[(e1[1],
                                                 e2[0])], vk_dict[(e1[1],
                                                                   e2[1])]
                    kn1 = nk11 * nk22
                    kn2 = nk12 * nk21
                    Kmatrix += kn1 + kn2

            # # ---- exact implementation of the Fast Computation of Shortest Path Kernel (FCSP), reference [2], sadly it is slower than the current implementation
            # # compute vertex kernel matrix
            # try:
            #     vk_mat = np.zeros((nx.number_of_nodes(g1),
            #                        nx.number_of_nodes(g2)))
            #     g1nl = enumerate(g1.nodes(data=True))
            #     g2nl = enumerate(g2.nodes(data=True))
            #     for i1, n1 in g1nl:
            #         for i2, n2 in g2nl:
            #             vk_mat[i1][i2] = kn(
            #                 n1[1][node_label], n2[1][node_label],
            #                 [n1[1]['attributes']], [n2[1]['attributes']])

            #     range1 = range(0, len(edge_w_g[i]))
            #     range2 = range(0, len(edge_w_g[j]))
            #     for i1 in range1:
            #         x1 = edge_x_g[i][i1]
            #         y1 = edge_y_g[i][i1]
            #         w1 = edge_w_g[i][i1]
            #         for i2 in range2:
            #             x2 = edge_x_g[j][i2]
            #             y2 = edge_y_g[j][i2]
            #             w2 = edge_w_g[j][i2]
            #             ke = (w1 == w2)
            #             if ke > 0:
            #                 kn1 = vk_mat[x1][x2] * vk_mat[y1][y2]
            #                 kn2 = vk_mat[x1][y2] * vk_mat[y1][x2]
            #                 Kmatrix += kn1 + kn2
    except KeyError:  # missing labels or attributes
        pass

    return i, j, Kmatrix


def wrap_getSPGraph(Gn, weight, i):
    return i, getSPGraph(Gn[i], edge_weight=weight)
    # return i, nx.floyd_warshall_numpy(Gn[i], weight=weight)


# def spkernel_do(Gn, ds_attrs, node_label, node_kernels, ij):

#     i = ij[0]
#     j = ij[1]
#     g1 = Gn[i]
#     g2 = Gn[j]
#     Kmatrix = 0
#     if ds_attrs['node_labeled']:
#         # node symb and non-synb labeled
#         if ds_attrs['node_attr_dim'] > 0:
#             if ds_attrs['is_directed']:
#                 for e1, e2 in product(
#                         Gn[i].edges(data=True), Gn[j].edges(data=True)):
#                     if e1[2]['cost'] == e2[2]['cost']:
#                         kn = node_kernels['mix']
#                         try:
#                             n11, n12, n21, n22 = Gn[i].nodes[e1[0]], Gn[
#                                 i].nodes[e1[1]], Gn[j].nodes[e2[0]], Gn[
#                                     j].nodes[e2[1]]
#                             kn1 = kn(
#                                 n11[node_label], n21[node_label],
#                                 [n11['attributes']], [n21['attributes']]) * kn(
#                                     n12[node_label], n22[node_label],
#                                     [n12['attributes']], [n22['attributes']])
#                             Kmatrix += kn1
#                         except KeyError:  # missing labels or attributes
#                             pass
#             else:
#                 kn = node_kernels['mix']
#                 try:
#                     # compute shortest path matrices first, method borrowed from FCSP.
#                     vk_dict = {}  # shortest path matrices dict
#                     for n1 in g1.nodes(data=True):
#                         for n2 in g2.nodes(data=True):
#                             vk_dict[(n1[0], n2[0])] = kn(
#                                 n1[1][node_label], n2[1][node_label],
#                                 [n1[1]['attributes']], [n2[1]['attributes']])

#                     for e1, e2 in product(
#                             g1.edges(data=True), g2.edges(data=True)):
#                         if e1[2]['cost'] == e2[2]['cost']:
#                             # each edge walk is counted twice, starting from both its extreme nodes.
#                             nk11, nk12, nk21, nk22 = vk_dict[(
#                                 e1[0],
#                                 e2[0])], vk_dict[(e1[0], e2[1])], vk_dict[(
#                                     e1[1], e2[0])], vk_dict[(e1[1], e2[1])]
#                             kn1 = nk11 * nk22
#                             kn2 = nk12 * nk21
#                             Kmatrix += kn1 + kn2

#                 # # ---- exact implementation of the Fast Computation of Shortest Path Kernel (FCSP), reference [2], sadly it is slower than the current implementation
#                 # # compute vertex kernel matrix
#                 # try:
#                 #     vk_mat = np.zeros((nx.number_of_nodes(g1),
#                 #                        nx.number_of_nodes(g2)))
#                 #     g1nl = enumerate(g1.nodes(data=True))
#                 #     g2nl = enumerate(g2.nodes(data=True))
#                 #     for i1, n1 in g1nl:
#                 #         for i2, n2 in g2nl:
#                 #             vk_mat[i1][i2] = kn(
#                 #                 n1[1][node_label], n2[1][node_label],
#                 #                 [n1[1]['attributes']], [n2[1]['attributes']])

#                 #     range1 = range(0, len(edge_w_g[i]))
#                 #     range2 = range(0, len(edge_w_g[j]))
#                 #     for i1 in range1:
#                 #         x1 = edge_x_g[i][i1]
#                 #         y1 = edge_y_g[i][i1]
#                 #         w1 = edge_w_g[i][i1]
#                 #         for i2 in range2:
#                 #             x2 = edge_x_g[j][i2]
#                 #             y2 = edge_y_g[j][i2]
#                 #             w2 = edge_w_g[j][i2]
#                 #             ke = (w1 == w2)
#                 #             if ke > 0:
#                 #                 kn1 = vk_mat[x1][x2] * vk_mat[y1][y2]
#                 #                 kn2 = vk_mat[x1][y2] * vk_mat[y1][x2]
#                 #                 Kmatrix += kn1 + kn2

#                 except KeyError:  # missing labels or attributes
#                     pass

#         # node symb labeled
#         else:
#             if ds_attrs['is_directed']:
#                 for e1, e2 in product(
#                         Gn[i].edges(data=True), Gn[j].edges(data=True)):
#                     if e1[2]['cost'] == e2[2]['cost']:
#                         kn = node_kernels['symb']
#                         try:
#                             n11, n12, n21, n22 = Gn[i].nodes[e1[0]], Gn[
#                                 i].nodes[e1[1]], Gn[j].nodes[e2[0]], Gn[
#                                     j].nodes[e2[1]]
#                             kn1 = kn(n11[node_label], n21[node_label]) * kn(
#                                 n12[node_label], n22[node_label])
#                             Kmatrix += kn1
#                         except KeyError:  # missing labels
#                             pass
#             else:
#                 kn = node_kernels['symb']
#                 try:
#                     # compute shortest path matrices first, method borrowed from FCSP.
#                     vk_dict = {}  # shortest path matrices dict
#                     for n1 in g1.nodes(data=True):
#                         for n2 in g2.nodes(data=True):
#                             vk_dict[(n1[0], n2[0])] = kn(
#                                 n1[1][node_label], n2[1][node_label])

#                     for e1, e2 in product(
#                             g1.edges(data=True), g2.edges(data=True)):
#                         if e1[2]['cost'] == e2[2]['cost']:
#                             # each edge walk is counted twice, starting from both its extreme nodes.
#                             nk11, nk12, nk21, nk22 = vk_dict[(
#                                 e1[0],
#                                 e2[0])], vk_dict[(e1[0], e2[1])], vk_dict[(
#                                     e1[1], e2[0])], vk_dict[(e1[1], e2[1])]
#                             kn1 = nk11 * nk22
#                             kn2 = nk12 * nk21
#                             Kmatrix += kn1 + kn2
#                 except KeyError:  # missing labels
#                     pass
#     else:
#         # node non-synb labeled
#         if ds_attrs['node_attr_dim'] > 0:
#             if ds_attrs['is_directed']:
#                 for e1, e2 in product(
#                         Gn[i].edges(data=True), Gn[j].edges(data=True)):
#                     if e1[2]['cost'] == e2[2]['cost']:
#                         kn = node_kernels['nsymb']
#                         try:
#                             # each edge walk is counted twice, starting from both its extreme nodes.
#                             n11, n12, n21, n22 = Gn[i].nodes[e1[0]], Gn[
#                                 i].nodes[e1[1]], Gn[j].nodes[e2[0]], Gn[
#                                     j].nodes[e2[1]]
#                             kn1 = kn(
#                                 [n11['attributes']], [n21['attributes']]) * kn(
#                                     [n12['attributes']], [n22['attributes']])
#                             Kmatrix += kn1
#                         except KeyError:  # missing attributes
#                             pass
#             else:
#                 for e1, e2 in product(
#                         Gn[i].edges(data=True), Gn[j].edges(data=True)):
#                     if e1[2]['cost'] == e2[2]['cost']:
#                         kn = node_kernels['nsymb']
#                         try:
#                             # each edge walk is counted twice, starting from both its extreme nodes.
#                             n11, n12, n21, n22 = Gn[i].nodes[e1[0]], Gn[
#                                 i].nodes[e1[1]], Gn[j].nodes[e2[0]], Gn[
#                                     j].nodes[e2[1]]
#                             kn1 = kn(
#                                 [n11['attributes']], [n21['attributes']]) * kn(
#                                     [n12['attributes']], [n22['attributes']])
#                             kn2 = kn(
#                                 [n11['attributes']], [n22['attributes']]) * kn(
#                                     [n12['attributes']], [n21['attributes']])
#                             Kmatrix += kn1 + kn2
#                         except KeyError:  # missing attributes
#                             pass
#         # node unlabeled
#         else:
#             for e1, e2 in product(
#                     Gn[i].edges(data=True), Gn[j].edges(data=True)):
#                 if e1[2]['cost'] == e2[2]['cost']:
#                     Kmatrix += 1

#     return i, j, Kmatrix
