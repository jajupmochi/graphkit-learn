#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:40:52 2018

@author: ljia
"""
import sys
import numpy as np
import networkx as nx

sys.path.insert(0, "../")
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.model_selection_precomputed import compute_gram_matrices
from gklearn.kernels.structuralspKernel import structuralspkernel
from sklearn.model_selection import ParameterGrid

from libs import *
import multiprocessing
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct


if __name__ == "__main__":
    # load dataset.
    print('getting dataset and computing kernel distance matrix first...')
    ds_name = 'SYNTHETICnew'
    gkernel = 'structuralspkernel'
    dataset = '../datasets/SYNTHETICnew/SYNTHETICnew_A.txt'
    Gn, y_all = loadDataset(dataset)

    for G in Gn:
        G.graph['filename'] = 'graph' + str(G.graph['name']) + '.gxl'
    
    # compute/read Gram matrix and pair distances.
    mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
    sub_kernels = {'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}
    Kmatrix, run_time = structuralspkernel(Gn, node_label=None, edge_label=None,
                          node_kernels=sub_kernels, edge_kernels=sub_kernels,
                          parallel=None,  # parallel='imap_unordered', 
                          n_jobs=multiprocessing.cpu_count(), 
                          verbose=True)
    
    # normalization
    Kmatrix_diag = Kmatrix.diagonal().copy()
    for i in range(len(Kmatrix)):
        for j in range(i, len(Kmatrix)):
            Kmatrix[i][j] /= np.sqrt(Kmatrix_diag[i] * Kmatrix_diag[j])
            Kmatrix[j][i] = Kmatrix[i][j]
    
    np.savez('results/xp_fit_method/Kmatrix.' + ds_name + '.' + gkernel + '.gm', 
         Kmatrix=Kmatrix, run_time=run_time)
    
    print('complete!')