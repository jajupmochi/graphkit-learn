#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:02:28 2018

@author: ljia
"""

import functools
from libs import *
import multiprocessing

from gklearn.kernels.randomWalkKernel import randomwalkkernel
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct

import numpy as np


dslist = [
#    {'name': 'Alkane', 'dataset': '../datasets/Alkane/dataset.ds', 'task': 'regression',
#        'dataset_y': '../datasets/Alkane/dataset_boiling_point_names.txt'},  
#    # contains single node graph, node symb
#    {'name': 'Acyclic', 'dataset': '../datasets/acyclic/dataset_bps.ds',
#        'task': 'regression'},  # node symb
#    {'name': 'MAO', 'dataset': '../datasets/MAO/dataset.ds'}, # node/edge symb
#    {'name': 'PAH', 'dataset': '../datasets/PAH/dataset.ds'}, # unlabeled
#    {'name': 'MUTAG', 'dataset': '../datasets/MUTAG/MUTAG_A.txt'}, # node/edge symb
#    {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt'},
#    # node nsymb
#    {'name': 'ENZYMES', 'dataset': '../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
#    # node symb/nsymb
    {'name': 'AIDS', 'dataset': '../datasets/AIDS/AIDS_A.txt'}, # node symb/nsymb, edge symb
    {'name': 'NCI1', 'dataset': '../datasets/NCI1/NCI1_A.txt'}, # node symb
    {'name': 'NCI109', 'dataset': '../datasets/NCI109/NCI109_A.txt'}, # node symb   
    {'name': 'D&D', 'dataset': '../datasets/DD/DD_A.txt'}, # node symb

#
#    {'name': 'Mutagenicity', 'dataset': '../datasets/Mutagenicity/Mutagenicity_A.txt'},
#    # node/edge symb
    #     {'name': 'COIL-DEL', 'dataset': '../datasets/COIL-DEL/COIL-DEL_A.txt'}, # edge symb, node nsymb
    # # #     {'name': 'BZR', 'dataset': '../datasets/BZR_txt/BZR_A_sparse.txt'}, # node symb/nsymb
    # # #     {'name': 'COX2', 'dataset': '../datasets/COX2_txt/COX2_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'Fingerprint', 'dataset': '../datasets/Fingerprint/Fingerprint_A.txt'},
    #
    # #     {'name': 'DHFR', 'dataset': '../datasets/DHFR_txt/DHFR_A_sparse.txt'}, # node symb/nsymb
    # #     {'name': 'SYNTHETIC', 'dataset': '../datasets/SYNTHETIC_txt/SYNTHETIC_A_sparse.txt'}, # node symb/nsymb
    # #     {'name': 'MSRC9', 'dataset': '../datasets/MSRC_9_txt/MSRC_9_A.txt'}, # node symb
    # #     {'name': 'MSRC21', 'dataset': '../datasets/MSRC_21_txt/MSRC_21_A.txt'}, # node symb
    # #     {'name': 'FIRSTMM_DB', 'dataset': '../datasets/FIRSTMM_DB/FIRSTMM_DB_A.txt'}, # node symb/nsymb ,edge nsymb

    # #     {'name': 'PROTEINS', 'dataset': '../datasets/PROTEINS_txt/PROTEINS_A_sparse.txt'}, # node symb/nsymb
    # #     {'name': 'PROTEINS_full', 'dataset': '../datasets/PROTEINS_full_txt/PROTEINS_full_A_sparse.txt'}, # node symb/nsymb
    #     {'name': 'NCI-HIV', 'dataset': '../datasets/NCI-HIV/AIDO99SD.sdf',
    #         'dataset_y': '../datasets/NCI-HIV/aids_conc_may04.txt',}, # node/edge symb

    #     # not working below
    #     {'name': 'PTC_FM', 'dataset': '../datasets/PTC/Train/FM.ds',},
    #     {'name': 'PTC_FR', 'dataset': '../datasets/PTC/Train/FR.ds',},
    #     {'name': 'PTC_MM', 'dataset': '../datasets/PTC/Train/MM.ds',},
    #     {'name': 'PTC_MR', 'dataset': '../datasets/PTC/Train/MR.ds',},
]
estimator = randomwalkkernel
param_grid = [{'C': np.logspace(-10, 10, num=41, base=10)},
              {'alpha': np.logspace(-10, 10, num=41, base=10)}]


## for non-symbolic labels.
#gkernels = [functools.partial(gaussiankernel, gamma=1 / ga) 
#            for ga in np.logspace(0, 10, num=11, base=10)]
#mixkernels = [functools.partial(kernelproduct, deltakernel, gk) for gk in gkernels]
#sub_kernels = [{'symb': deltakernel, 'nsymb': gkernels[i], 'mix': mixkernels[i]}
#                for i in range(len(gkernels))]

# for symbolic labels only.
#gaussiankernel = functools.partial(gaussiankernel, gamma=0.5)
mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
sub_kernels = [{'symb': deltakernel, 'nsymb': gaussiankernel, 'mix': mixkernel}]

for ds in dslist:
    print()
    print(ds['name'])
    for compute_method in ['sylvester', 'conjugate', 'fp', 'spectral']:
#    for compute_method in ['conjugate', 'fp']:
        if compute_method == 'sylvester':
            param_grid_precomputed = {'compute_method': ['sylvester'],
#                          'weight': np.linspace(0.01, 0.10, 10)}
                          'weight': np.logspace(-1, -10, num=10, base=10)}
        elif compute_method == 'conjugate':
            mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
            param_grid_precomputed = {'compute_method': ['conjugate'], 
                          'node_kernels': sub_kernels, 'edge_kernels': sub_kernels,
                          'weight': np.logspace(-1, -10, num=10, base=10)}
        elif compute_method == 'fp':
            mixkernel = functools.partial(kernelproduct, deltakernel, gaussiankernel)
            param_grid_precomputed = {'compute_method': ['fp'], 
                          'node_kernels': sub_kernels, 'edge_kernels': sub_kernels,
                          'weight': np.logspace(-3, -10, num=8, base=10)}
        elif compute_method == 'spectral':
            param_grid_precomputed = {'compute_method': ['spectral'],
                          'weight': np.logspace(-1, -10, num=10, base=10),
                          'sub_kernel': ['geo', 'exp']}
        model_selection_for_precomputed_kernel(
            ds['dataset'],
            estimator,
            param_grid_precomputed,
            (param_grid[1] if ('task' in ds and ds['task']
                               == 'regression') else param_grid[0]),
            (ds['task'] if 'task' in ds else 'classification'),
            NUM_TRIALS=30,
            datafile_y=(ds['dataset_y'] if 'dataset_y' in ds else None),
            extra_params=(ds['extra_params'] if 'extra_params' in ds else None),
            ds_name=ds['name'],
            n_jobs=multiprocessing.cpu_count(),
            read_gm_from_file=False,
            verbose=True)
    print()
