#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:48:06 2019

@author: ljia
"""

import sys
import numpy as np
import networkx as nx

sys.path.insert(0, "../../")
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.model_selection_precomputed import compute_gram_matrices
from sklearn.model_selection import ParameterGrid

from libs import *
import multiprocessing
import functools
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct

dslist = [
    {'name': 'ENZYMES', 'dataset': '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    # node symb/nsymb
]

def run_ms(dataset, y, ds):
    from gklearn.kernels.randomWalkKernel import randomwalkkernel
    estimator = randomwalkkernel
    param_grid = [{'C': np.logspace(-10, 10, num=41, base=10)},
              {'alpha': np.logspace(-10, 10, num=41, base=10)}]
    ave_time = {}
    std_time = {}
    for compute_method in ['sylvester', 'conjugate', 'fp', 'spectral']:
        if compute_method == 'sylvester':
            param_grid_precomputed = {'compute_method': ['sylvester'],
#                          'weight': np.linspace(0.01, 0.10, 10)}
                          'weight': np.logspace(-1, -10, num=10, base=10)}
        elif compute_method == 'conjugate':
            mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
            param_grid_precomputed = {'compute_method': ['conjugate'], 
                          'node_kernels': 
                          [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
                          'edge_kernels': 
                          [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
                          'weight': np.logspace(-1, -10, num=10, base=10)}
        elif compute_method == 'fp':
            mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
            param_grid_precomputed = {'compute_method': ['fp'], 
                          'node_kernels': 
                          [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
                          'edge_kernels': 
                          [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}],
                          'weight': np.logspace(-3, -10, num=8, base=10)}
        elif compute_method == 'spectral':
            param_grid_precomputed = {'compute_method': ['spectral'],
                          'weight': np.logspace(-1, -10, num=10, base=10),
                          'sub_kernel': ['geo', 'exp']}
        _, gram_matrix_time, _, _, _ = compute_gram_matrices(
                dataset, y, estimator, list(ParameterGrid(param_grid_precomputed)),
                '../../notebooks/results/' + estimator.__name__, ds['name'],
                n_jobs=multiprocessing.cpu_count(), verbose=False)
        average_gram_matrix_time = np.mean(gram_matrix_time)
        std_gram_matrix_time = np.std(gram_matrix_time, ddof=1)
        print('\n***** time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s'
              .format(average_gram_matrix_time, std_gram_matrix_time))
        ave_time[compute_method] = average_gram_matrix_time
        std_time[compute_method] = std_gram_matrix_time
    print()
    return ave_time, std_time


for ds in dslist:
    print()
    print(ds['name'])
    Gn, y_all = loadDataset(
        ds['dataset'], filename_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
        extra_params=(ds['extra_params'] if 'extra_params' in ds else None))
    degree_list = [np.mean(list(dict(g.degree()).values())) for g in Gn]
    idx_sorted = np.argsort(degree_list)
    degree_list.sort()
    Gn = [Gn[idx] for idx in idx_sorted]
    y_all = [y_all[idx] for idx in idx_sorted]
    len_1piece = int(len(Gn) / 5)
    ave_time = []
    std_time = []
    ave_degree = []
    for piece in range(0, 5):
        print('piece', str(piece), ':')
        Gn_p = Gn[len_1piece * piece:len_1piece * (piece + 1)]
        y_all_p = y_all[len_1piece * piece:len_1piece * (piece + 1)]
        aved = np.mean(degree_list[len_1piece * piece:len_1piece * (piece + 1)])
        ave_degree.append(aved)
        avet, stdt = run_ms(Gn_p, y_all_p, ds)
        ave_time.append(avet)
        std_time.append(stdt)
        
        print('\n****** for dataset', ds['name'], ', the average time is \n', ave_time,
              '\nthe time std is \n', std_time)
        print('corresponding average vertex degrees are', ave_degree)
    print()