#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:56:44 2018

@author: ljia
"""

import functools
from libs import *
import multiprocessing

from gklearn.kernels.rwalk_sym import randomwalkkernel
from gklearn.utils.kernels import deltakernel, gaussiankernel, kernelproduct

import numpy as np


dslist = [
    {'name': 'Letter-med', 'dataset': '../datasets/Letter-med/Letter-med_A.txt'},
    # node nsymb
    {'name': 'ENZYMES', 'dataset': '../datasets/ENZYMES_txt/ENZYMES_A_sparse.txt'},
    # node symb/nsymb
]
estimator = randomwalkkernel
param_grid = [{'C': np.logspace(-10, 10, num=41, base=10)},
              {'alpha': np.logspace(-10, 10, num=41, base=10)}]

for ds in dslist:
    print()
    print(ds['name'])
    for compute_method in ['conjugate', 'fp']:
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
            read_gm_from_file=False)
    print()