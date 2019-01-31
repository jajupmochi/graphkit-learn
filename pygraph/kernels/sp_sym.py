#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:02:00 2018

@author: ljia
"""

import sys
import time
from itertools import product
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
import numpy as np

from pygraph.utils.utils import getSPGraph
from pygraph.utils.graphdataset import get_dataset_attributes
from pygraph.utils.parallel import parallel_gm
sys.path.insert(0, "../")

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
    node_label : string
        node attribute used as label. The default node label is atom.
    edge_weight : string
        Edge attribute name corresponding to the edge weight.
    node_kernels: dict
        A dictionary of kernel functions for nodes, including 3 items: 'symb' 
        for symbolic node labels, 'nsymb' for non-symbolic node labels, 'mix' 
        for both labels. The first 2 functions take two node labels as 
        parameters, and the 'mix' function takes 4 parameters, a symbolic and a
        non-symbolic label for each the two nodes. Each label is in form of 2-D
        dimension array (n_samples, n_features). Each function returns an 
        number as the kernel value. Ignored when nodes are unlabeled.

    Return
    ------
    Kmatrix : Numpy matrix
        Kernel matrix, each element of which is the sp kernel between 2 praphs.
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
        attr_names=['node_labeled', 'node_attr_dim', 'is_directed'],
        node_label=node_label)
    ds_attrs['node_attr_dim'] = 0

    # remove graphs with no edges, as no sp can be found in their structures, 
    # so the kernel between such a graph and itself will be zero.
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
    getsp_partial = partial(wrapper_getSPGraph, weight)
    itr = zip(Gn, range(0, len(Gn)))
    if len(Gn) < 100 * n_jobs:
#        # use default chunksize as pool.map when iterable is less than 100
#        chunksize, extra = divmod(len(Gn), n_jobs * 4)
#        if extra:
#            chunksize += 1
        chunksize = int(len(Gn) / n_jobs) + 1
    else:
        chunksize = 100
    for i, g in tqdm(
            pool.imap_unordered(getsp_partial, itr, chunksize),
            desc='getting sp graphs', file=sys.stdout):
        Gn[i] = g
    pool.close()
    pool.join()

    Kmatrix = np.zeros((len(Gn), len(Gn)))

    # ---- use pool.imap_unordered to parallel and track progress. ----
    def init_worker(gn_toshare):
        global G_gn
        G_gn = gn_toshare
    do_partial = partial(wrapper_sp_do, ds_attrs, node_label, node_kernels)   
    parallel_gm(do_partial, Kmatrix, Gn, init_worker=init_worker, 
                glbv=(Gn,), n_jobs=n_jobs)

    run_time = time.time() - start_time
    print(
        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time, idx


def spkernel_do(g1, g2, ds_attrs, node_label, node_kernels):
    
    kernel = 0

    # compute shortest path matrices first, method borrowed from FCSP.
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
            for e1, e2 in product(
                    g1.edges(data=True), g2.edges(data=True)):
                if e1[2]['cost'] == e2[2]['cost']:
                    kernel += 1
            return kernel

    # compute graph kernels
    if ds_attrs['is_directed']:
        for e1, e2 in product(g1.edges(data=True), g2.edges(data=True)):
            if e1[2]['cost'] == e2[2]['cost']:
                nk11, nk22 = vk_dict[(e1[0], e2[0])], vk_dict[(e1[1],
                                                               e2[1])]
                kn1 = nk11 * nk22
                kernel += kn1
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
                kernel += kn1 + kn2

    return kernel


def wrapper_sp_do(ds_attrs, node_label, node_kernels, itr):
    i = itr[0]
    j = itr[1]
    return i, j, spkernel_do(G_gn[i], G_gn[j], ds_attrs, node_label, node_kernels)


def wrapper_getSPGraph(weight, itr_item):
    g = itr_item[0]
    i = itr_item[1]
    return i, getSPGraph(g, edge_weight=weight)