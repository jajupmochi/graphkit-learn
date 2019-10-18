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

import sys
sys.path.insert(0, "../")
from pygraph.kernels.marginalizedKernel import marginalizedkernel
from pygraph.kernels.untilHPathKernel import untilhpathkernel
from pygraph.kernels.spKernel import spkernel
import functools
from pygraph.utils.kernels import deltakernel, gaussiankernel, kernelproduct
from pygraph.kernels.structuralspKernel import structuralspkernel


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


def compute_kernel(Gn, graph_kernel, verbose):
    if graph_kernel == 'marginalizedkernel':
        Kmatrix, _ = marginalizedkernel(Gn, node_label='atom', edge_label=None,
                                  p_quit=0.03, n_iteration=10, remove_totters=False,
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'untilhpathkernel':
        Kmatrix, _ = untilhpathkernel(Gn, node_label='atom', edge_label=None,
                                  depth=10, k_func='MinMax', compute_method='trie',
                                  n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'spkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix, _, _ = spkernel(Gn, node_label='atom', node_kernels=
                              {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
                              n_jobs=multiprocessing.cpu_count(), verbose=verbose)
    elif graph_kernel == 'structuralspkernel':
        mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
        Kmatrix, _ = structuralspkernel(Gn, node_label='atom', node_kernels=
                              {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel},
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


def kernel_distance_matrix(Gn, Kmatrix=None, gkernel=None):
    dis_mat = np.empty((len(Gn), len(Gn)))
    if Kmatrix == None:
        Kmatrix = compute_kernel(Gn, gkernel, True)
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