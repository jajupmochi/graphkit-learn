#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:57:18 2018

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

dslist = [
#    {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
#        'task': 'regression'},  # node symb
#    {'name': 'Alkane', 'dataset': '../datasets/Alkane/dataset.ds', 'task': 'regression',
#             'dataset_y': '../datasets/Alkane/dataset_boiling_point_names.txt', },  
#    # contains single node graph, node symb
#    {'name': 'MAO', 'dataset': '../datasets/MAO/dataset.ds', },  # node/edge symb
    {'name': 'PAH', 'dataset': '../../datasets/PAH/dataset.ds', },  # unlabeled
    {'name': 'MUTAG', 'dataset': '../../datasets/MUTAG/MUTAG.mat',
             'extra_params': {'am_sp_al_nl_el': [0, 0, 3, 1, 2]}},  # node/edge symb
#    {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt'},
#    # node nsymb
    {'name': 'ENZYMES', 'dataset': '../../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    # node symb/nsymb
]

def run_ms(dataset, y, ds):
    from gklearn.kernels.commonWalkKernel import commonwalkkernel
    estimator = commonwalkkernel
    param_grid_precomputed = [{'compute_method': ['geo'], 
                               'weight': np.linspace(0.01, 0.15, 15)},
    #                           'weight': np.logspace(-1, -10, num=10, base=10)},
                              {'compute_method': ['exp'], 'weight': range(0, 15)}]
    param_grid = [{'C': np.logspace(-10, 10, num=41, base=10)},
                  {'alpha': np.logspace(-10, 10, num=41, base=10)}]

    _, gram_matrix_time, _, _, _ = compute_gram_matrices(
                dataset, y, estimator, list(ParameterGrid(param_grid_precomputed)),
                '../../notebooks/results/' + estimator.__name__, ds['name'],
                n_jobs=multiprocessing.cpu_count(), verbose=False)
    average_gram_matrix_time = np.mean(gram_matrix_time)
    std_gram_matrix_time = np.std(gram_matrix_time, ddof=1)
    print('\n***** time to calculate gram matrix with different hyper-params: {:.2f}±{:.2f}s'
          .format(average_gram_matrix_time, std_gram_matrix_time))
    print()
    return average_gram_matrix_time, std_gram_matrix_time


for ds in dslist:
    print()
    print(ds['name'])
    Gn, y_all = loadDataset(
        ds['dataset'], filename_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
        extra_params=(ds['extra_params'] if 'extra_params' in ds else None))
    vn_list = [nx.number_of_nodes(g) for g in Gn]
    idx_sorted = np.argsort(vn_list)
    vn_list.sort()
    Gn = [Gn[idx] for idx in idx_sorted]
    y_all = [y_all[idx] for idx in idx_sorted]
    len_1piece = int(len(Gn) / 5)
    ave_time = []
    std_time = []
    ave_vnb = []
    for piece in range(0, 5):
        print('piece', str(piece), ':')
        Gn_p = Gn[len_1piece * piece:len_1piece * (piece + 1)]
        y_all_p = y_all[len_1piece * piece:len_1piece * (piece + 1)]
        avevn = np.mean(vn_list[len_1piece * piece:len_1piece * (piece + 1)])
        ave_vnb.append(avevn)
        avet, stdt = run_ms(Gn_p, y_all_p, ds)
        ave_time.append(avet)
        std_time.append(stdt)
        
        print('\n****** for dataset', ds['name'], ', the average time is \n', ave_time,
              '\nthe time std is \n', std_time)
        print('corresponding average vertex numbers are', ave_vnb)
    print()