#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:42:48 2018

@author: ljia
"""

import sys
import time
from itertools import combinations, product
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
import numpy as np

from pygraph.utils.graphdataset import get_dataset_attributes
from pygraph.utils.parallel import parallel_gm

sys.path.insert(0, "../")


def structuralspkernel(*args,
                       node_label='atom',
                       edge_weight=None,
                       edge_label='bond_type',
                       node_kernels=None,
                       edge_kernels=None,
                       n_jobs=None):
    """Calculate mean average structural shortest path kernels between graphs.

    Parameters
    ----------
    Gn : List of NetworkX graph
        List of graphs between which the kernels are calculated.
    /
    G1, G2 : NetworkX graphs
        2 graphs between which the kernel is calculated.
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_weight : string
        Edge attribute name corresponding to the edge weight.
    edge_label : string
        edge attribute used as label. The default edge label is bond_type.
    node_kernels: dict
        A dictionary of kernel functions for nodes, including 3 items: 'symb' 
        for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix' 
        for both labels. The first 2 functions take two node labels as 
        parameters, and the 'mix' function takes 4 parameters, a symbolic and a
        non-symbolic label for each the two nodes. Each label is in form of 2-D
        dimension array (n_samples, n_features). Each function returns a number
        as the kernel value. Ignored when nodes are unlabeled.
    edge_kernels: dict
        A dictionary of kernel functions for edges, including 3 items: 'symb' 
        for symbolic edge labels, 'nsymb' for non-symbolic edge labels, 'mix' 
        for both labels. The first 2 functions take two edge labels as 
        parameters, and the 'mix' function takes 4 parameters, a symbolic and a
        non-symbolic label for each the two edges. Each label is in form of 2-D
        dimension array (n_samples, n_features). Each function returns a number
        as the kernel value. Ignored when edges are unlabeled.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the mean average structural 
        shortest path kernel between 2 praphs.
    """
    # pre-process
    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
    weight = None
    if edge_weight is None:
        print('\n None edge weight specified. Set all weight to 1.\n')
    else:
        try:
            some_weight = list(
                nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
            if isinstance(some_weight, (float, int)):
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
        attr_names=['node_labeled', 'node_attr_dim', 'edge_labeled',
                    'edge_attr_dim', 'is_directed'],
        node_label=node_label, edge_label=edge_label)
    ds_attrs['node_attr_dim'] = 0
    ds_attrs['edge_attr_dim'] = 0

    start_time = time.time()

    # get shortest paths of each graph in Gn
    splist = [None] * len(Gn)
    pool = Pool(n_jobs)
    # get shortest path graphs of Gn
    getsp_partial = partial(wrapper_getSP, weight, ds_attrs['is_directed'])
    itr = zip(Gn, range(0, len(Gn)))
    if len(Gn) < 100 * n_jobs:
        chunksize = int(len(Gn) / n_jobs) + 1
    else:
        chunksize = 100
    # chunksize = 300  # int(len(list(itr)) / n_jobs)
    for i, sp in tqdm(
            pool.imap_unordered(getsp_partial, itr, chunksize),
            desc='getting shortest paths',
            file=sys.stdout):
        splist[i] = sp
#        time.sleep(10)
    pool.close()
    pool.join()
    
    
#    # get shortest paths of each graph in Gn
#    splist = [[] for _ in range(len(Gn))]
#    # get shortest path graphs of Gn
#    getsp_partial = partial(wrapper_getSP, weight, ds_attrs['is_directed'])
#    itr = zip(Gn, range(0, len(Gn)))
#    if len(Gn) < 1000 * n_jobs:
#        chunksize = int(len(Gn) / n_jobs) + 1
#    else:
#        chunksize = 1000
#    # chunksize = 300  # int(len(list(itr)) / n_jobs)
#    from contextlib import closing  
#    with closing(Pool(n_jobs)) as pool:
##        for i, sp in tqdm(
#        res = pool.imap_unordered(getsp_partial, itr, 10)
##                desc='getting shortest paths',
##                file=sys.stdout):
##            splist[i] = sp
##        time.sleep(10)
#    pool.close()
#    pool.join()
    
#    ss = 0
#    ss += sys.getsizeof(splist)
#    for spss in splist:
#        ss += sys.getsizeof(spss)
#        for spp in spss:
#            ss += sys.getsizeof(spp)
    
    
#    time.sleep(20)
    
#    # ---- direct running, normally use single CPU core. ----
#    splist = []
#    for g in tqdm(Gn, desc='getting sp graphs', file=sys.stdout):
#        splist.append(get_shortest_paths(g, weight, ds_attrs['is_directed']))

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
    def init_worker(spl_toshare, gs_toshare):
        global G_spl, G_gs
        G_spl = spl_toshare
        G_gs = gs_toshare      
    do_partial = partial(wrapper_ssp_do, ds_attrs, node_label, edge_label, 
                         node_kernels, edge_kernels)   
    parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
                        glbv=(splist, Gn), n_jobs=n_jobs) 
    

#    # ---- use pool.imap_unordered to parallel and track progress. ----
#    pool = Pool(n_jobs)
#    do_partial = partial(wrapper_ssp_do, ds_attrs, node_label, edge_label, 
#                         node_kernels, edge_kernels)
#    itr = zip(combinations_with_replacement(Gn, 2),
#              combinations_with_replacement(splist, 2),
#              combinations_with_replacement(range(0, len(Gn)), 2))
#    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
#    if len_itr < 1000 * n_jobs:
#        chunksize = int(len_itr / n_jobs) + 1
#    else:
#        chunksize = 1000
#    for i, j, kernel in tqdm(
#            pool.imap_unordered(do_partial, itr, chunksize),
#            desc='calculating kernels',
#            file=sys.stdout):
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel
#    pool.close()
#    pool.join()
    
#    # ---- use pool.map to parallel. ----
#    pool = Pool(n_jobs)
#    do_partial = partial(wrapper_ssp_do, ds_attrs, node_label, edge_label, 
#                         node_kernels, edge_kernels)
#    itr = zip(combinations_with_replacement(Gn, 2),
#              combinations_with_replacement(splist, 2),
#              combinations_with_replacement(range(0, len(Gn)), 2))
#    for i, j, kernel in tqdm(
#            pool.map(do_partial, itr), desc='calculating kernels',
#            file=sys.stdout):
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel
#    pool.close()
#    pool.join()

#    # ---- use pool.imap_unordered to parallel and track progress. ----
#    do_partial = partial(wrapper_ssp_do, ds_attrs, node_label, edge_label, 
#                         node_kernels, edge_kernels)
#    itr = zip(combinations_with_replacement(Gn, 2),
#              combinations_with_replacement(splist, 2),
#              combinations_with_replacement(range(0, len(Gn)), 2))
#    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
#    if len_itr < 1000 * n_jobs:
#        chunksize = int(len_itr / n_jobs) + 1
#    else:
#        chunksize = 1000
#    from contextlib import closing
#    with closing(Pool(n_jobs)) as pool:
#        for i, j, kernel in tqdm(
#                pool.imap_unordered(do_partial, itr, 1000),
#                desc='calculating kernels',
#                file=sys.stdout):
#            Kmatrix[i][j] = kernel
#            Kmatrix[j][i] = kernel
#    pool.close()
#    pool.join()


#    # ---- direct running, normally use single CPU core. ----
#    from itertools import combinations_with_replacement
#    itr = combinations_with_replacement(range(0, len(Gn)), 2)
#    for i, j in tqdm(itr, desc='calculating kernels', file=sys.stdout):
#        kernel = structuralspkernel_do(Gn[i], Gn[j], splist[i], splist[j],
#                ds_attrs, node_label, edge_label, node_kernels, edge_kernels)
##        if(kernel > 1):
##            print("error here ")
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel

    run_time = time.time() - start_time
    print(
        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time


def structuralspkernel_do(g1, g2, spl1, spl2, ds_attrs, node_label, edge_label,
                          node_kernels, edge_kernels):
    
    kernel = 0

    # First, compute shortest path matrices, method borrowed from FCSP.
    vk_dict = {}  # shortest path matrices dict
    if ds_attrs['node_labeled']:
        # node symb and non-synb labeled
        if ds_attrs['node_attr_dim'] > 0:
            kn = node_kernels['mix']
            for n1, n2 in product(
                    g1.nodes(data=True), g2.nodes(data=True)):
                vk_dict[(n1[0], n2[0])] = kn(
                    n1[1][node_label], n2[1][node_label],
                    n1[1]['attributes'], n2[1]['attributes'])
        # node symb labeled
        else:
            kn = node_kernels['symb']
            for n1 in g1.nodes(data=True):
                for n2 in g2.nodes(data=True):
                    vk_dict[(n1[0], n2[0])] = kn(n1[1][node_label],
                                                 n2[1][node_label])
    else:
        # node non-synb labeled
        if ds_attrs['node_attr_dim'] > 0:
            kn = node_kernels['nsymb']
            for n1 in g1.nodes(data=True):
                for n2 in g2.nodes(data=True):
                    vk_dict[(n1[0], n2[0])] = kn(n1[1]['attributes'],
                                                 n2[1]['attributes'])
        # node unlabeled
        else:
            pass

    # Then, compute kernels between all pairs of edges, which idea is an
    # extension of FCSP. It suits sparse graphs, which is the most case we
    # went though. For dense graphs, this would be slow.
    ek_dict = {}  # dict of edge kernels
    if ds_attrs['edge_labeled']:
        # edge symb and non-synb labeled
        if ds_attrs['edge_attr_dim'] > 0:
            ke = edge_kernels['mix']
            for e1, e2 in product(
                    g1.edges(data=True), g2.edges(data=True)):
                ek_temp = ke(e1[2][edge_label], e2[2][edge_label],
                    e1[2]['attributes'], e2[2]['attributes'])
                ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
                ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
                ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
                ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
        # edge symb labeled
        else:
            ke = edge_kernels['symb']
            for e1 in g1.edges(data=True):
                for e2 in g2.edges(data=True):
                    ek_temp = ke(e1[2][edge_label], e2[2][edge_label])
                    ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
                    ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
                    ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
                    ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
    else:
        # edge non-synb labeled
        if ds_attrs['edge_attr_dim'] > 0:
            ke = edge_kernels['nsymb']
            for e1 in g1.edges(data=True):
                for e2 in g2.edges(data=True):
                    ek_temp = kn(e1[2]['attributes'], e2[2]['attributes'])
                    ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ek_temp
                    ek_dict[((e1[1], e1[0]), (e2[0], e2[1]))] = ek_temp
                    ek_dict[((e1[0], e1[1]), (e2[1], e2[0]))] = ek_temp
                    ek_dict[((e1[1], e1[0]), (e2[1], e2[0]))] = ek_temp
        # edge unlabeled
        else:
            pass

    # compute graph kernels
    if vk_dict:
        if ek_dict:
            for p1, p2 in product(spl1, spl2):
                if len(p1) == len(p2):
                    kpath = vk_dict[(p1[0], p2[0])]
                    if kpath:
                        for idx in range(1, len(p1)):
                            kpath *= vk_dict[(p1[idx], p2[idx])] * \
                                ek_dict[((p1[idx-1], p1[idx]),
                                         (p2[idx-1], p2[idx]))]
                            if not kpath:
                                break
                        kernel += kpath  # add up kernels of all paths
        else:
            for p1, p2 in product(spl1, spl2):
                if len(p1) == len(p2):
                    kpath = vk_dict[(p1[0], p2[0])]
                    if kpath:
                        for idx in range(1, len(p1)):
                            kpath *= vk_dict[(p1[idx], p2[idx])]
                            if not kpath:
                                break
                        kernel += kpath  # add up kernels of all paths
    else:
        if ek_dict:
            for p1, p2 in product(spl1, spl2):
                if len(p1) == len(p2):
                    if len(p1) == 0:
                        kernel += 1
                    else:
                        kpath = 1
                        for idx in range(0, len(p1) - 1):
                            kpath *= ek_dict[((p1[idx], p1[idx+1]),
                                              (p2[idx], p2[idx+1]))]
                            if not kpath:
                                break
                        kernel += kpath  # add up kernels of all paths
        else:
            for p1, p2 in product(spl1, spl2):
                if len(p1) == len(p2):
                    kernel += 1

    kernel = kernel / (len(spl1) * len(spl2))  # calculate mean average

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
    return kernel


def wrapper_ssp_do(ds_attrs, node_label, edge_label, node_kernels, 
                   edge_kernels, itr):
    i = itr[0]
    j = itr[1]
    return i, j, structuralspkernel_do(G_gs[i], G_gs[j], G_spl[i], G_spl[j], 
                                       ds_attrs, node_label, edge_label, 
                                       node_kernels, edge_kernels)


def get_shortest_paths(G, weight, directed):
    """Get all shortest paths of a graph.

    Parameters
    ----------
    G : NetworkX graphs
        The graphs whose paths are calculated.
    weight : string/None
        edge attribute used as weight to calculate the shortest path.
    directed: boolean
        Whether graph is directed.

    Return
    ------
    sp : list of list
        List of shortest paths of the graph, where each path is represented by a list of nodes.
    """
    sp = []
    for n1, n2 in combinations(G.nodes(), 2):
        try:
            spltemp = list(nx.all_shortest_paths(G, n1, n2, weight=weight))
        except nx.NetworkXNoPath:  # nodes not connected
            #            sp.append([])
            pass
        else:
            sp += spltemp
            # each edge walk is counted twice, starting from both its extreme nodes.
            if not directed:
                sp += [sptemp[::-1] for sptemp in spltemp]
                
    # add single nodes as length 0 paths.
    sp += [[n] for n in G.nodes()]
    return sp


def wrapper_getSP(weight, directed, itr_item):
    g = itr_item[0]
    i = itr_item[1]
    return i, get_shortest_paths(g, weight, directed)