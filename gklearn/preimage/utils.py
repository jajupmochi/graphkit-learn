#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:05:07 2019

Useful functions.
@author: ljia
"""
#import networkx as nx

import multiprocessing
import numpy as np

from gklearn.kernels.marginalizedKernel import marginalizedkernel
from gklearn.kernels.untilHPathKernel import untilhpathkernel
from gklearn.kernels.spKernel import spkernel
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct, polynomialkernel
from gklearn.kernels.structuralspKernel import structuralspkernel
from gklearn.kernels.treeletKernel import treeletkernel
from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel


def remove_edges(Gn):
    for G in Gn:
        for _, _, attrs in G.edges(data=True):
            attrs.clear()
            
def dis_gstar(idx_g, idx_gi, alpha, Kmatrix, term3=0, withterm3=True):
    term1 = Kmatrix[idx_g, idx_g]
    term2 = 0
    for i, a in enumerate(alpha):
        term2 += a * Kmatrix[idx_g, idx_gi[i]]
    term2 *= 2
    if withterm3 == False:
        for i1, a1 in enumerate(alpha):
            for i2, a2 in enumerate(alpha):
                term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    return np.sqrt(term1 - term2 + term3)


def compute_kernel(Gn, graph_kernel, node_label, edge_label, verbose, parallel='imap_unordered'):
    if graph_kernel == 'marginalizedkernel':
        Kmatrix, _ = marginalizedkernel(Gn, node_label=node_label, edge_label=edge_label,
                                  p_quit=0.03, n_iteration=10, remove_totters=False,
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'untilhpathkernel':
        Kmatrix, _ = untilhpathkernel(Gn, node_label=node_label, edge_label=edge_label,
                                  depth=7, k_func='MinMax', compute_method='trie',
                                  parallel=parallel,
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'spkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix = np.empty((len(Gn), len(Gn)))
#        Kmatrix[:] = np.nan
        Kmatrix, _, idx = spkernel(Gn, node_label=node_label, node_kernels=
                              {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
                              n_jobs=multiprocessing.cpu_count(), verbose=verbose)
#        for i, row in enumerate(idx):
#            for j, col in enumerate(idx):
#                Kmatrix[row, col] = Kmatrix_tmp[i, j]
    elif graph_kernel == 'structuralspkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
        Kmatrix, _ = structuralspkernel(Gn, node_label=node_label, 
                              edge_label=edge_label, node_kernels=sub_kernels,
                              edge_kernels=sub_kernels,
                              parallel=parallel, n_jobs=multiprocessing.cpu_count(), 
                              verbose=verbose)
    elif graph_kernel == 'treeletkernel':
        pkernel = functools.partial(polynomialkernel, d=2, c=1e5)
#        pkernel = functools.partial(gaussiankernel, gamma=1e-6)
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix, _ = treeletkernel(Gn, node_label=node_label, edge_label=edge_label,
                                   sub_kernel=pkernel, parallel=parallel,
                                   n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'weisfeilerlehmankernel':
        Kmatrix, _ = weisfeilerlehmankernel(Gn, node_label=node_label, edge_label=edge_label,
                                   height=4, base_kernel='subtree', parallel=None,
                                   n_jobs=multiprocessing.cpu_count(), verbose=verbose)
        
    # normalization
    Kmatrix_diag = Kmatrix.diagonal().copy()
    for i in range(len(Kmatrix)):
        for j in range(i, len(Kmatrix)):
            Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
            Kmatrix[j][i] = Kmatrix[i][j]
    return Kmatrix
            

def gram2distances(Kmatrix):
    dmatrix = np.zeros((len(Kmatrix), len(Kmatrix)))
    for i1 in range(len(Kmatrix)):
        for i2 in range(len(Kmatrix)):
            dmatrix[i1, i2] = Kmatrix[i1, i1] + Kmatrix[i2, i2] - 2 * Kmatrix[i1, i2]
    dmatrix = np.sqrt(dmatrix)
    return dmatrix


def kernel_distance_matrix(Gn, node_label, edge_label, Kmatrix=None, 
                           gkernel=None, verbose=True):
    dis_mat = np.empty((len(Gn), len(Gn)))
    if Kmatrix is None:
        Kmatrix = compute_kernel(Gn, gkernel, node_label, edge_label, verbose)
    for i in range(len(Gn)):
        for j in range(i, len(Gn)):
            dis = Kmatrix[i, i] + Kmatrix[j, j] - 2 * Kmatrix[i, j]
            if dis < 0:
                if dis > -1e-10:
                    dis = 0
                else:
                    raise ValueError('The distance is negative.')
            dis_mat[i, j] = np.sqrt(dis)
            dis_mat[j, i] = dis_mat[i, j]
    dis_max = np.max(np.max(dis_mat))
    dis_min = np.min(np.min(dis_mat[dis_mat != 0]))
    dis_mean = np.mean(np.mean(dis_mat))
    return dis_mat, dis_max, dis_min, dis_mean


def get_same_item_indices(ls):
    """Get the indices of the same items in a list. Return a dict keyed by items.
    """
    idx_dict = {}
    for idx, item in enumerate(ls):
        if item in idx_dict:
            idx_dict[item].append(idx)
        else:
            idx_dict[item] = [idx]
    return idx_dict


def k_nearest_neighbors_to_median_in_kernel_space(Gn, Kmatrix=None, gkernel=None,
                                                  node_label=None, edge_label=None):
    dis_k_all = [] # distance between g_star and each graph.
    alpha = [1 / len(Gn)] * len(Gn)
    if Kmatrix is None:
        Kmatrix = compute_kernel(Gn, gkernel, node_label, edge_label, True)
    term3 = 0
    for i1, a1 in enumerate(alpha):
        for i2, a2 in enumerate(alpha):
            term3 += a1 * a2 * Kmatrix[idx_gi[i1], idx_gi[i2]]
    for ig, g in tqdm(enumerate(Gn_init), desc='computing distances', file=sys.stdout):
        dtemp = dis_gstar(ig, idx_gi, alpha, Kmatrix, term3=term3)
        dis_all.append(dtemp)


def normalize_distance_matrix(D):
    max_value = np.amax(D)
    min_value = np.amin(D)
    return (D - min_value) / (max_value - min_value)