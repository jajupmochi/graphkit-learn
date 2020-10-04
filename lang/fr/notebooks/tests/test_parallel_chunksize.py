#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test of parallel, find the best parallel chunksize and iteration seperation scheme.
Created on Wed Sep 26 12:09:34 2018

@author: ljia
"""

import sys
import time
from itertools import combinations_with_replacement, product, combinations
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx
import numpy as np
import functools
#import multiprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from libs import *
from gklearn.utils.utils import getSPGraph, direct_product
from gklearn.utils.graphdataset import get_dataset_attributes
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct


def spkernel(*args,
             node_label='atom',
             edge_weight=None,
             node_kernels=None,
             n_jobs=None,
             chunksize=1):
    """Calculate shortest-path kernels between graphs.
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
    itr = combinations_with_replacement(range(0, len(Gn)), 2)
    with Pool(processes=n_jobs, initializer=init_worker, initargs=(Gn,)) as pool:
        for i, j, kernel in tqdm(pool.imap_unordered(do_partial, itr, chunksize),
                                 desc='calculating kernels', file=sys.stdout):
            Kmatrix[i][j] = kernel
            Kmatrix[j][i] = kernel
    
#    # ---- direct running, normally use single CPU core. ----
#    itr = combinations_with_replacement(range(0, len(Gn)), 2)
#    for i, j in tqdm(itr, desc='calculating kernels', file=sys.stdout):
#        kernel = spkernel_do(Gn[i], Gn[j], ds_attrs, node_label, node_kernels)
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel

    run_time = time.time() - start_time
    print(
        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
        % (len(Gn), run_time))

    return Kmatrix, run_time, idx


def spkernel_do(g1, g2, ds_attrs, node_label, node_kernels):
    
    kernel = 0

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
                    n1[1]['attributes'], n2[1]['attributes'])
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



#
#
#def commonwalkkernel(*args,
#                     node_label='atom',
#                     edge_label='bond_type',
#                     n=None,
#                     weight=1,
#                     compute_method=None,
#                     n_jobs=None,
#                     chunksize=1):
#    """Calculate common walk graph kernels between graphs.
#    """
#    compute_method = compute_method.lower()
#    # arrange all graphs in a list
#    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
#    Kmatrix = np.zeros((len(Gn), len(Gn)))
#    ds_attrs = get_dataset_attributes(
#        Gn,
#        attr_names=['node_labeled', 'edge_labeled', 'is_directed'],
#        node_label=node_label,
#        edge_label=edge_label)
#    if not ds_attrs['node_labeled']:
#        for G in Gn:
#            nx.set_node_attributes(G, '0', 'atom')
#    if not ds_attrs['edge_labeled']:
#        for G in Gn:
#            nx.set_edge_attributes(G, '0', 'bond_type')
#    if not ds_attrs['is_directed']:  # convert
#        Gn = [G.to_directed() for G in Gn]
#
#    start_time = time.time()
#
#    # ---- use pool.imap_unordered to parallel and track progress. ----
#    pool = Pool(n_jobs)
#    itr = combinations_with_replacement(range(0, len(Gn)), 2)
##    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
##    if len_itr < 100:
##        chunksize, extra = divmod(len_itr, n_jobs * 4)
##        if extra:
##            chunksize += 1
##    else:
##        chunksize = 100
#
#    # direct product graph method - exponential
#    if compute_method == 'exp':
#        do_partial = partial(_commonwalkkernel_exp, Gn, node_label, edge_label,
#                             weight)
#    # direct product graph method - geometric
#    elif compute_method == 'geo':
#        do_partial = partial(_commonwalkkernel_geo, Gn, node_label, edge_label,
#                             weight)
#
#    for i, j, kernel in tqdm(
#            pool.imap_unordered(do_partial, itr, chunksize),
#            desc='calculating kernels',
#            file=sys.stdout):
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel
#    pool.close()
#    pool.join()
#
#    run_time = time.time() - start_time
#    print(
#        "\n --- kernel matrix of common walk kernel of size %d built in %s seconds ---"
#        % (len(Gn), run_time))
#
#    return Kmatrix, run_time
#
#
#def _commonwalkkernel_exp(Gn, node_label, edge_label, beta, ij):
#    """Calculate walk graph kernels up to n between 2 graphs using exponential 
#    series.
#    """
#    i = ij[0]
#    j = ij[1]
#    g1 = Gn[i]
#    g2 = Gn[j]
#
#    # get tensor product / direct product
#    gp = direct_product(g1, g2, node_label, edge_label)
#    A = nx.adjacency_matrix(gp).todense()
#
#    ew, ev = np.linalg.eig(A)
#    D = np.zeros((len(ew), len(ew)))
#    for i in range(len(ew)):
#        D[i][i] = np.exp(beta * ew[i])
#    exp_D = ev * D * ev.T
#
#    return i, j, exp_D.sum()
#
#
#def _commonwalkkernel_geo(Gn, node_label, edge_label, gamma, ij):
#    """Calculate common walk graph kernels up to n between 2 graphs using 
#    geometric series.
#    """
#    i = ij[0]
#    j = ij[1]
#    g1 = Gn[i]
#    g2 = Gn[j]
#
#    # get tensor product / direct product
#    gp = direct_product(g1, g2, node_label, edge_label)
#    A = nx.adjacency_matrix(gp).todense()
#    mat = np.identity(len(A)) - gamma * A
#    try:
#        return i, j, mat.I.sum()
#    except np.linalg.LinAlgError:
#        return i, j, np.nan


#def structuralspkernel(*args,
#                       node_label='atom',
#                       edge_weight=None,
#                       edge_label='bond_type',
#                       node_kernels=None,
#                       edge_kernels=None,
#                       n_jobs=None,
#                       chunksize=1):
#    """Calculate mean average structural shortest path kernels between graphs.
#    """
#    # pre-process
#    Gn = args[0] if len(args) == 1 else [args[0], args[1]]
#
#    weight = None
#    if edge_weight is None:
#        print('\n None edge weight specified. Set all weight to 1.\n')
#    else:
#        try:
#            some_weight = list(
#                nx.get_edge_attributes(Gn[0], edge_weight).values())[0]
#            if isinstance(some_weight, (float, int)):
#                weight = edge_weight
#            else:
#                print(
#                    '\n Edge weight with name %s is not float or integer. Set all weight to 1.\n'
#                    % edge_weight)
#        except:
#            print(
#                '\n Edge weight with name "%s" is not found in the edge attributes. Set all weight to 1.\n'
#                % edge_weight)
#    ds_attrs = get_dataset_attributes(
#        Gn,
#        attr_names=['node_labeled', 'node_attr_dim', 'edge_labeled',
#                    'edge_attr_dim', 'is_directed'],
#        node_label=node_label, edge_label=edge_label)
#
#    start_time = time.time()
#
#    # get shortest paths of each graph in Gn
#    splist = [[] for _ in range(len(Gn))]
#    pool = Pool(n_jobs)
#    # get shortest path graphs of Gn
#    getsp_partial = partial(wrap_getSP, Gn, weight, ds_attrs['is_directed'])
##    if len(Gn) < 100:
##        # use default chunksize as pool.map when iterable is less than 100
##        chunksize, extra = divmod(len(Gn), n_jobs * 4)
##        if extra:
##            chunksize += 1
##    else:
##        chunksize = 100
#    # chunksize = 300  # int(len(list(itr)) / n_jobs)
#    for i, sp in tqdm(
#            pool.imap_unordered(getsp_partial, range(0, len(Gn)), chunksize),
#            desc='getting shortest paths',
#            file=sys.stdout):
#        splist[i] = sp
#
#    Kmatrix = np.zeros((len(Gn), len(Gn)))
#
#    # ---- use pool.imap_unordered to parallel and track progress. ----
#    do_partial = partial(structuralspkernel_do, Gn, splist, ds_attrs,
#                         node_label, edge_label, node_kernels, edge_kernels)
#    itr = combinations_with_replacement(range(0, len(Gn)), 2)
##    len_itr = int(len(Gn) * (len(Gn) + 1) / 2)
##    if len_itr < 100:
##        chunksize, extra = divmod(len_itr, n_jobs * 4)
##        if extra:
##            chunksize += 1
##    else:
##        chunksize = 100
#    for i, j, kernel in tqdm(
#            pool.imap_unordered(do_partial, itr, chunksize),
#            desc='calculating kernels',
#            file=sys.stdout):
#        Kmatrix[i][j] = kernel
#        Kmatrix[j][i] = kernel
#    pool.close()
#    pool.join()
#
#    run_time = time.time() - start_time
#    print(
#        "\n --- shortest path kernel matrix of size %d built in %s seconds ---"
#        % (len(Gn), run_time))
#
#    return Kmatrix, run_time
#
#
#def structuralspkernel_do(Gn, splist, ds_attrs, node_label, edge_label,
#                          node_kernels, edge_kernels, ij):
#
#    iglobal = ij[0]
#    jglobal = ij[1]
#    g1 = Gn[iglobal]
#    g2 = Gn[jglobal]
#    spl1 = splist[iglobal]
#    spl2 = splist[jglobal]
#    kernel = 0
#
#    try:
#        # First, compute shortest path matrices, method borrowed from FCSP.
#        if ds_attrs['node_labeled']:
#            # node symb and non-synb labeled
#            if ds_attrs['node_attr_dim'] > 0:
#                kn = node_kernels['mix']
#                vk_dict = {}  # shortest path matrices dict
#                for n1, n2 in product(
#                        g1.nodes(data=True), g2.nodes(data=True)):
#                    vk_dict[(n1[0], n2[0])] = kn(
#                        n1[1][node_label], n2[1][node_label],
#                        [n1[1]['attributes']], [n2[1]['attributes']])
#            # node symb labeled
#            else:
#                kn = node_kernels['symb']
#                vk_dict = {}  # shortest path matrices dict
#                for n1 in g1.nodes(data=True):
#                    for n2 in g2.nodes(data=True):
#                        vk_dict[(n1[0], n2[0])] = kn(n1[1][node_label],
#                                                     n2[1][node_label])
#        else:
#            # node non-synb labeled
#            if ds_attrs['node_attr_dim'] > 0:
#                kn = node_kernels['nsymb']
#                vk_dict = {}  # shortest path matrices dict
#                for n1 in g1.nodes(data=True):
#                    for n2 in g2.nodes(data=True):
#                        vk_dict[(n1[0], n2[0])] = kn([n1[1]['attributes']],
#                                                     [n2[1]['attributes']])
#            # node unlabeled
#            else:
#                vk_dict = {}
#
#        # Then, compute kernels between all pairs of edges, which idea is an
#        # extension of FCSP. It suits sparse graphs, which is the most case we
#        # went though. For dense graphs, it would be slow.
#        if ds_attrs['edge_labeled']:
#            # edge symb and non-synb labeled
#            if ds_attrs['edge_attr_dim'] > 0:
#                ke = edge_kernels['mix']
#                ek_dict = {}  # dict of edge kernels
#                for e1, e2 in product(
#                        g1.edges(data=True), g2.edges(data=True)):
#                    ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ke(
#                        e1[2][edge_label], e2[2][edge_label],
#                        [e1[2]['attributes']], [e2[2]['attributes']])
#            # edge symb labeled
#            else:
#                ke = edge_kernels['symb']
#                ek_dict = {}
#                for e1 in g1.edges(data=True):
#                    for e2 in g2.edges(data=True):
#                        ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = ke(
#                            e1[2][edge_label], e2[2][edge_label])
#        else:
#            # edge non-synb labeled
#            if ds_attrs['edge_attr_dim'] > 0:
#                ke = edge_kernels['nsymb']
#                ek_dict = {}
#                for e1 in g1.edges(data=True):
#                    for e2 in g2.edges(data=True):
#                        ek_dict[((e1[0], e1[1]), (e2[0], e2[1]))] = kn(
#                            [e1[2]['attributes']], [e2[2]['attributes']])
#            # edge unlabeled
#            else:
#                ek_dict = {}
#
#        # compute graph kernels
#        if vk_dict:
#            if ek_dict:
#                for p1, p2 in product(spl1, spl2):
#                    if len(p1) == len(p2):
#                        kpath = vk_dict[(p1[0], p2[0])]
#                        if kpath:
#                            for idx in range(1, len(p1)):
#                                kpath *= vk_dict[(p1[idx], p2[idx])] * \
#                                    ek_dict[((p1[idx-1], p1[idx]),
#                                             (p2[idx-1], p2[idx]))]
#                                if not kpath:
#                                    break
#                            kernel += kpath  # add up kernels of all paths
#            else:
#                for p1, p2 in product(spl1, spl2):
#                    if len(p1) == len(p2):
#                        kpath = vk_dict[(p1[0], p2[0])]
#                        if kpath:
#                            for idx in range(1, len(p1)):
#                                kpath *= vk_dict[(p1[idx], p2[idx])]
#                                if not kpath:
#                                    break
#                            kernel += kpath  # add up kernels of all paths
#        else:
#            if ek_dict:
#                for p1, p2 in product(spl1, spl2):
#                    if len(p1) == len(p2):
#                        if len(p1) == 0:
#                            kernel += 1
#                        else:
#                            kpath = 1
#                            for idx in range(0, len(p1) - 1):
#                                kpath *= ek_dict[((p1[idx], p1[idx+1]),
#                                                  (p2[idx], p2[idx+1]))]
#                                if not kpath:
#                                    break
#                            kernel += kpath  # add up kernels of all paths
#            else:
#                for p1, p2 in product(spl1, spl2):
#                    if len(p1) == len(p2):
#                        kernel += 1
#
#        kernel = kernel / (len(spl1) * len(spl2))  # calculate mean average
#    except KeyError:  # missing labels or attributes
#        pass
#
#    return iglobal, jglobal, kernel
#
#
#def get_shortest_paths(G, weight, directed):
#    """Get all shortest paths of a graph.
#    """
#    sp = []
#    for n1, n2 in combinations(G.nodes(), 2):
#        try:
#            sptemp = nx.shortest_path(G, n1, n2, weight=weight)
#            sp.append(sptemp)
#            # each edge walk is counted twice, starting from both its extreme nodes.
#            if not directed:
#                sp.append(sptemp[::-1])
#        except nx.NetworkXNoPath:  # nodes not connected
#            #            sp.append([])
#            pass
#    # add single nodes as length 0 paths.
#    sp += [[n] for n in G.nodes()]
#    return sp
#
#
#def wrap_getSP(Gn, weight, directed, i):
#    return i, get_shortest_paths(Gn[i], weight, directed)


def compute_gram_matrices(datafile,
                          estimator,
                          param_grid_precomputed,
                          datafile_y=None,
                          extra_params=None,
                          ds_name='ds-unknown',
                          n_jobs=1,
                          chunksize=1):
    """

    Parameters
    ----------
    datafile : string
        Path of dataset file.
    estimator : function
        kernel function used to estimate. This function needs to return a gram matrix.
    param_grid_precomputed : dictionary
        Dictionary with names (string) of parameters used to calculate gram matrices as keys and lists of parameter settings to try as values. This enables searching over any sequence of parameter settings. Params with length 1 will be omitted.
    datafile_y : string
        Path of file storing y data. This parameter is optional depending on the given dataset file.
    """
    tqdm.monitor_interval = 0

    # Load the dataset
    dataset, y_all = loadDataset(
        datafile, filename_y=datafile_y, extra_params=extra_params)

    # Grid of parameters with a discrete number of values for each.
    param_list_precomputed = list(ParameterGrid(param_grid_precomputed))

    gram_matrix_time = [
    ]  # a list to store time to calculate gram matrices

    # calculate all gram matrices
    for idx, params_out in enumerate(param_list_precomputed):
        y = y_all[:]
        params_out['n_jobs'] = n_jobs
        params_out['chunksize'] = chunksize
        rtn_data = estimator(dataset[:], **params_out)
        Kmatrix = rtn_data[0]
        current_run_time = rtn_data[1]
        # for some kernels, some graphs in datasets may not meet the
        # kernels' requirements for graph structure. These graphs are trimmed.
        if len(rtn_data) == 3:
            idx_trim = rtn_data[2]  # the index of trimmed graph list
            y = [y[idx] for idx in idx_trim]  # trim y accordingly

        Kmatrix_diag = Kmatrix.diagonal().copy()
        # remove graphs whose kernels with themselves are zeros
        nb_g_ignore = 0
        for idx, diag in enumerate(Kmatrix_diag):
            if diag == 0:
                Kmatrix = np.delete(Kmatrix, (idx - nb_g_ignore), axis=0)
                Kmatrix = np.delete(Kmatrix, (idx - nb_g_ignore), axis=1)
                nb_g_ignore += 1
        # normalization
        for i in range(len(Kmatrix)):
            for j in range(i, len(Kmatrix)):
                Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
                Kmatrix[j][i] = Kmatrix[i][j]

        gram_matrix_time.append(current_run_time)

    average_gram_matrix_time = np.mean(gram_matrix_time)

    return average_gram_matrix_time


dslist = [
    {'name': 'Acyclic', 'dataset': '../../datasets/acyclic/dataset_bps.ds',
        'task': 'regression'},  # node symb
    {'name': 'Alkane', 'dataset': '../../datasets/Alkane/dataset.ds', 'task': 'regression',
             'dataset_y': '../../datasets/Alkane/dataset_boiling_point_names.txt', },  # contains single node graph, node symb
    {'name': 'MAO', 'dataset': '../../datasets/MAO/dataset.ds', },  # node/edge symb
    {'name': 'PAH', 'dataset': '../../datasets/PAH/dataset.ds', },  # unlabeled
    {'name': 'MUTAG', 'dataset': '../../datasets/MUTAG/MUTAG.mat',
             'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}},  # node/edge symb
    {'name': 'Letter-med', 'dataset': '../../datasets/Letter-med/Letter-med_A.txt'},
    # node symb/nsymb
    {'name': 'ENZYMES', 'dataset': '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    # node/edge symb
#    {'name': 'Mutagenicity', 'dataset': '../../datasets/Mutagenicity/Mutagenicity_A.txt'},
#    {'name': 'D&D', 'dataset': '../../datasets/D&D/DD.mat',
#     'extra_params': {'am_sp_al_nl_el': [0, 1, 2, 1, -1]}},  # node symb
]

fig, ax = plt.subplots()
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlabel('parallel chunksize')
ax.set_ylabel('runtime($s$)')
ax.set_title('28 cpus')
ax.grid(axis='both')

estimator = spkernel
if estimator.__name__ == 'spkernel':
    mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
    param_grid_precomputed = {'node_kernels': [
        {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]}

elif estimator.__name__ == 'commonwalkkernel':
    mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
    param_grid_precomputed = {'compute_method': ['geo'],
                               'weight': [1]}           
elif estimator.__name__ == 'structuralspkernel':
    mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
    param_grid_precomputed = {'node_kernels': 
        [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
        'edge_kernels': 
        [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]}                 

#list(range(10, 100, 20)) + 
#chunklist = list(range(10, 100, 20)) + list(range(100, 1000, 200)) + \
#    list(range(1000, 10000, 2000)) + list(range(10000, 100000, 20000))
#    chunklist = list(range(300, 1000, 200)) + list(range(1000, 10000, 2000)) + list(range(10000, 100000, 20000))
chunklist = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + \
    list(range(1000, 10000, 1000)) + list(range(10000, 100000, 10000))
#chunklist = list(range(1000, 10000, 1000))
gmtmat = np.zeros((len(dslist), len(chunklist)))
cpus = 28   

for idx1, ds in enumerate(dslist):
    print()
    print(ds['name'])

    for idx2, cs in enumerate(chunklist):
        print(ds['name'], idx2, cs)
        gmtmat[idx1][idx2] = compute_gram_matrices(
            ds['dataset'],
            estimator,
            param_grid_precomputed,

            datafile_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
            extra_params=(ds['extra_params']
                          if 'extra_params' in ds else None),
            ds_name=ds['name'],
            n_jobs=cpus,
            chunksize=cs)

    print()
    print(gmtmat[idx1, :])
    np.save('../test_parallel/' + estimator.__name__ + '.' + ds['name'] + '_' + 
            str(idx1), gmtmat[idx1, :])

    p = ax.plot(chunklist, gmtmat[idx1, :], '.-', label=ds['name'], zorder=3)    
    ax.legend(loc='upper right', ncol=3, labelspacing=0.1, handletextpad=0.4, 
              columnspacing=0.6)
    plt.savefig('../test_parallel/' + estimator.__name__ + str(idx1) + '_' + 
                str(cpus) + '.eps', format='eps', dpi=300)
#    plt.show()